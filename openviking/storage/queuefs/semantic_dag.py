# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Semantic DAG executor with event-driven lazy dispatch."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from openviking.server.identity import RequestContext
from openviking.storage.queuefs.semantic_sidecar import write_semantic_sidecars
from openviking.storage.transaction import NO_LOCK, LockLease, get_lock_manager
from openviking.storage.viking_fs import get_viking_fs
from openviking.telemetry.request_wait_tracker import get_request_wait_tracker
from openviking_cli.utils import VikingURI
from openviking_cli.utils.logger import get_logger

logger = get_logger(__name__)

# Session-internal files that should never be summarized by the semantic pipeline.
# These are canonical archives (e.g. session transcripts) whose content provides
# no additional retrieval value and would only waste tokens and add latency.
_SKIP_FILENAMES = frozenset({"messages.jsonl"})


@dataclass
class DirNode:
    """Directory node state for DAG execution."""

    uri: str
    children_dirs: List[str]
    file_paths: List[str]
    file_index: Dict[str, int]
    child_index: Dict[str, int]
    file_summaries: List[Optional[Dict[str, str]]]
    children_abstracts: List[Optional[Dict[str, str]]]
    pending: int
    dispatched: bool = False
    overview_scheduled: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class DagStats:
    total_nodes: int = 0
    pending_nodes: int = 0
    in_progress_nodes: int = 0
    done_nodes: int = 0


@dataclass
class VectorizeTask:
    """Vectorize task information."""

    task_type: str  # "file" or "directory"
    uri: str
    context_type: str
    ctx: "RequestContext"
    semantic_msg_id: Optional[str] = None
    # For file tasks
    file_path: Optional[str] = None
    summary_dict: Optional[Dict[str, str]] = None
    parent_uri: Optional[str] = None
    use_summary: bool = False
    # Cached text content captured during semantic summary so the embedding
    # stage does not have to re-read the same file from AGFS (1 HEAD + 1 GET
    # per file on S3). Only populated for text files.
    prefetched_text: Optional[str] = None
    # For directory tasks
    abstract: Optional[str] = None
    overview: Optional[str] = None


class SemanticDagExecutor:
    """Execute semantic generation with DAG-style, event-driven lazy dispatch."""

    def __init__(
        self,
        processor: "SemanticProcessor",
        context_type: str,
        max_concurrent_llm: int,
        ctx: RequestContext,
        incremental_update: bool = False,
        target_uri: Optional[str] = None,
        semantic_msg_id: Optional[str] = None,
        telemetry_id: str = "",
        recursive: bool = True,
        lock: LockLease = NO_LOCK,
        is_code_repo: bool = False,
        changes: Optional[Dict[str, List[str]]] = None,
        skip_vectorization: bool = False,
        coalesce_key: str = "",
        coalesce_version: int = 0,
        write_concurrency: int = 8,
    ):
        self._processor = processor
        self._context_type = context_type
        self._max_concurrent_llm = max_concurrent_llm
        self._ctx = ctx
        self._incremental_update = incremental_update
        self._target_uri = target_uri
        self._semantic_msg_id = semantic_msg_id
        self._telemetry_id = telemetry_id
        self._recursive = recursive
        self._lock = lock
        self._is_code_repo = is_code_repo
        self._changes = changes or {}
        self._skip_vectorization = skip_vectorization
        self._coalesce_key = coalesce_key
        self._coalesce_version = coalesce_version
        self._stale = False
        self._changed_paths = {
            path for key in ("added", "modified", "deleted") for path in self._changes.get(key, [])
        }
        self._llm_sem = asyncio.Semaphore(max_concurrent_llm)
        # Bound how many directories flush their sidecars to storage at once.
        # Locks are now cheap (temp-tree pre-acquire), so all sibling dirs would
        # otherwise burst their writes concurrently and throttle S3 Express.
        self._write_sem = asyncio.Semaphore(max(1, write_concurrency))
        self._viking_fs = get_viking_fs()
        self._nodes: Dict[str, DirNode] = {}
        self._parent: Dict[str, Optional[str]] = {}
        self._root_uri: Optional[str] = None
        self._root_done: Optional[asyncio.Event] = None
        self._stats = DagStats()
        self._vectorize_task_count: int = 0
        self._pending_vectorize_tasks: List[VectorizeTask] = []
        self._vectorize_lock = asyncio.Lock()
        self._file_change_status: Dict[str, bool] = {}
        self._dir_change_status: Dict[str, bool] = {}
        self._overview_cache: Dict[str, Dict[str, str]] = {}
        self._overview_cache_lock = asyncio.Lock()
        # Perf instrumentation: aggregate the (uninstrumented) bottom-up directory
        # overview roll-up so the post-summary silent gap is visible at a glance.
        self._rollup_t0: Optional[float] = None
        self._overview_acc: Dict[str, float] = {
            "dirs": 0,
            "lock": 0.0,
            "llm": 0.0,
            "read_existing": 0.0,
            "children": 0.0,
            "write": 0.0,
            "unchanged": 0,
            "generated": 0,
        }

    async def run(self, root_uri: str) -> None:
        """Run DAG execution starting from root_uri."""
        self._root_uri = root_uri
        self._root_done = asyncio.Event()

        # Pre-acquire a TREE lock over the temp root for the duration of the
        # bottom-up overview roll-up. Each per-directory write_semantic_sidecars
        # takes a LockContext exact-acquire on temp sidecar paths; without an
        # owned ancestor tree there, acquire does a serial S3 ancestor walk
        # (measured ~20-35s/dir on Express — the dominant cost of the
        # post-summary gap). Registering this tree lock on the handle populates
        # lock_types so those acquires hit the in-memory cache. Released once
        # the roll-up finishes — before SyncDiff (on_complete) re-acquires its
        # own temp tree lock. Mirrors SyncDiff Change 1 in semantic_processor.
        temp_tree_locks: List[str] = []
        handle = getattr(self._lock, "handle", None)
        if handle is not None:
            try:
                root_path = self._viking_fs._uri_to_path(root_uri, ctx=self._ctx)
                locks_before = set(handle.locks)
                if await get_lock_manager().acquire_tree(handle, root_path):
                    temp_tree_locks = [
                        lp for lp in handle.locks if lp not in locks_before
                    ]
                else:
                    logger.warning(
                        "[overview] temp tree pre-acquire returned False for %s; "
                        "roll-up uses slow ancestor walk",
                        root_path,
                    )
            except Exception as e:
                logger.warning("[overview] temp tree pre-acquire failed: %s", e)

        async def _release_temp_tree_locks() -> None:
            nonlocal temp_tree_locks
            if not temp_tree_locks:
                return
            to_release, temp_tree_locks = temp_tree_locks, []
            try:
                await get_lock_manager().release_selected(handle, to_release)
            except Exception as e:
                logger.warning("[overview] failed to release temp tree lock: %s", e)

        try:
            await self._dispatch_dir(root_uri, parent_uri=None)
            await self._root_done.wait()
        except Exception:
            await self._lock.close()
            raise
        finally:
            await _release_temp_tree_locks()

        # Release owned semantic locks after downstream vectorization finishes.
        async def wrapped_on_complete() -> None:
            try:
                if self._telemetry_id and self._semantic_msg_id:
                    get_request_wait_tracker().mark_semantic_done(
                        self._telemetry_id, self._semantic_msg_id
                    )
            finally:
                await self._lock.close()

        async with self._vectorize_lock:
            task_count = self._vectorize_task_count
            tasks = list(self._pending_vectorize_tasks)

        if task_count > 0:
            from .embedding_tracker import EmbeddingTaskTracker

            tracker = EmbeddingTaskTracker.get_instance()
            await tracker.register(
                semantic_msg_id=self._semantic_msg_id,
                total_count=task_count,
                on_complete=wrapped_on_complete,
                metadata={"uri": root_uri},
            )

            for task in tasks:
                if task.task_type == "file":
                    asyncio.create_task(
                        self._processor._vectorize_single_file(
                            parent_uri=task.parent_uri,
                            context_type=task.context_type,
                            file_path=task.file_path,
                            summary_dict=task.summary_dict,
                            ctx=task.ctx,
                            semantic_msg_id=task.semantic_msg_id,
                            use_summary=task.use_summary,
                            prefetched_text=task.prefetched_text,
                        )
                    )
                else:
                    asyncio.create_task(
                        self._processor._vectorize_directory(
                            task.uri,
                            task.context_type,
                            task.abstract,
                            task.overview,
                            ctx=task.ctx,
                            semantic_msg_id=task.semantic_msg_id,
                        )
                    )
        else:
            # No vectorize tasks — release lock immediately (via wrapped callback)
            try:
                await wrapped_on_complete()
            except Exception as e:
                logger.error(f"Error in on_complete callback: {e}", exc_info=True)

    async def _dispatch_dir(self, dir_uri: str, parent_uri: Optional[str]) -> None:
        """Lazy-dispatch tasks for a directory when it is triggered."""
        if dir_uri in self._nodes:
            return

        self._parent[dir_uri] = parent_uri

        try:
            children_dirs, file_paths = await self._list_dir(dir_uri, "_dispatch_dir")
            file_index = {path: idx for idx, path in enumerate(file_paths)}
            child_index = {path: idx for idx, path in enumerate(children_dirs)}
            if self._recursive:
                pending = len(children_dirs) + len(file_paths)
            else:
                pending = len(file_paths)

            node = DirNode(
                uri=dir_uri,
                children_dirs=children_dirs,
                file_paths=file_paths,
                file_index=file_index,
                child_index=child_index,
                file_summaries=[None] * len(file_paths),
                children_abstracts=[None] * len(children_dirs),
                pending=pending,
                dispatched=True,
            )
            self._nodes[dir_uri] = node
            self._stats.total_nodes += 1
            self._stats.pending_nodes += 1

            if pending == 0:
                self._schedule_overview(dir_uri)
                return

            for file_path in file_paths:
                self._stats.total_nodes += 1
                # File nodes are scheduled immediately: pending -> in_progress.
                self._stats.pending_nodes += 1
                self._stats.pending_nodes = max(0, self._stats.pending_nodes - 1)
                self._stats.in_progress_nodes += 1
                asyncio.create_task(self._file_summary_task(dir_uri, file_path))

            if children_dirs:
                if self._recursive:
                    for child_uri in children_dirs:
                        asyncio.create_task(self._dispatch_dir(child_uri, dir_uri))
        except Exception as e:
            logger.error(f"Failed to dispatch directory {dir_uri}: {e}", exc_info=True)
            if parent_uri:
                await self._on_child_done(parent_uri, dir_uri, "")
            elif self._root_done:
                self._root_done.set()

    async def _list_dir(self, uri: str, from_hint: str) -> tuple[list[str], list[str]]:
        """List directory entries and return (child_dirs, file_paths)."""
        try:
            entries = await self._viking_fs.ls(uri, ctx=self._ctx)
        except Exception as e:
            logger.warning(
                f"[SemanticDagExecutor] Failed to list directory {uri}: {e} from {from_hint}"
            )
            return [], []

        children_dirs: List[str] = []
        file_paths: List[str] = []

        for entry in entries:
            name = entry.get("name", "")
            if not name or name.startswith(".") or name in [".", ".."] or name in _SKIP_FILENAMES:
                continue

            item_uri = VikingURI(uri).join(name).uri
            if entry.get("isDir", False):
                children_dirs.append(item_uri)
            else:
                file_paths.append(item_uri)

        return children_dirs, file_paths

    def _get_target_file_path(self, current_uri: str) -> Optional[str]:
        if not self._incremental_update or not self._target_uri or not self._root_uri:
            logger.warning(
                f"Invalid target_uri or root_uri for incremental update: target_uri={self._target_uri}, root_uri={self._root_uri}"
            )
            return None
        if self._target_uri != self._root_uri:
            logger.warning(
                "Incremental semantic update expects target_uri == root_uri: "
                f"target_uri={self._target_uri}, root_uri={self._root_uri}"
            )
            return None
        return current_uri

    def _is_direct_incremental_update(self) -> bool:
        return (
            self._incremental_update
            and bool(self._changed_paths)
            and self._target_uri == self._root_uri
        )

    def _path_has_direct_change(self, uri: str) -> bool:
        if uri in self._changed_paths:
            return True
        prefix = uri.rstrip("/") + "/"
        return any(path.startswith(prefix) for path in self._changed_paths)

    async def _check_file_content_changed(self, file_path: str) -> bool:
        if self._is_direct_incremental_update():
            return file_path in self._changed_paths
        target_path = self._get_target_file_path(file_path)
        if not target_path:
            return True
        try:
            current_stat = await self._viking_fs.stat(file_path, ctx=self._ctx)
            target_stat = await self._viking_fs.stat(target_path, ctx=self._ctx)
            current_size = current_stat.get("size") if isinstance(current_stat, dict) else None
            target_size = target_stat.get("size") if isinstance(target_stat, dict) else None
            if current_size is not None and target_size is not None and current_size != target_size:
                return True
            # Use size + modTime as the primary heuristic (rsync -t style).
            # On S3, the previous content-equality check cost 1 HEAD + 1 GET
            # per side per file -- 400+ extra round trips for a 100-file
            # incremental update. Only fall back to full content compare when
            # modTime is unavailable on either side.
            current_mtime = (
                current_stat.get("modTime") if isinstance(current_stat, dict) else None
            )
            target_mtime = target_stat.get("modTime") if isinstance(target_stat, dict) else None
            if current_mtime is not None and target_mtime is not None:
                return current_mtime != target_mtime
            current_content = await self._viking_fs.read_file(file_path, ctx=self._ctx)
            target_content = await self._viking_fs.read_file(target_path, ctx=self._ctx)
            return current_content != target_content
        except Exception:
            return True

    async def _read_existing_summary(self, file_path: str) -> Optional[Dict[str, str]]:
        """Read existing summary from parent directory's .overview.md.

        Args:
            file_path: Current file path

        Returns:
            Summary dict with 'name' and 'summary' keys, or None if not found
        """
        target_path = self._get_target_file_path(file_path)
        if not target_path:
            return None

        try:
            parent_uri = "/".join(target_path.rsplit("/", 1)[:-1])
            if not parent_uri:
                return None

            if parent_uri not in self._overview_cache:
                try:
                    from openviking.metrics.datasources.cache import CacheEventDataSource

                    CacheEventDataSource.record_miss("L1")
                except Exception:
                    pass
                async with self._overview_cache_lock:
                    if parent_uri not in self._overview_cache:
                        overview_path = f"{parent_uri}/.overview.md"
                        overview_content = await self._viking_fs.read_file(
                            overview_path, ctx=self._ctx
                        )
                        if overview_content:
                            self._overview_cache[parent_uri] = self._processor._parse_overview_md(
                                overview_content
                            )
                        else:
                            self._overview_cache[parent_uri] = {}
            else:
                try:
                    from openviking.metrics.datasources.cache import CacheEventDataSource

                    CacheEventDataSource.record_hit("L1")
                except Exception:
                    pass

            existing_summaries = self._overview_cache.get(parent_uri, {})
            file_name = file_path.split("/")[-1]

            if file_name in existing_summaries:
                return {"name": file_name, "summary": existing_summaries[file_name]}

        except Exception as e:
            logger.debug(f"Failed to read existing summary from overview.md for {file_path}: {e}")

        return None

    async def _check_dir_children_changed(
        self, dir_uri: str, current_files: List[str], current_dirs: List[str]
    ) -> bool:
        if self._is_direct_incremental_update():
            if self._path_has_direct_change(dir_uri):
                return True
            for current_file in current_files:
                if self._file_change_status.get(current_file, True):
                    return True
            for current_dir in current_dirs:
                if self._dir_change_status.get(current_dir, True):
                    return True
            return False

        target_path = self._get_target_file_path(dir_uri)
        if not target_path:
            return True
        try:
            target_dirs, target_files = await self._list_dir(
                target_path, "_check_dir_children_changed"
            )
            current_file_names = {f.split("/")[-1] for f in current_files}
            target_file_names = {f.split("/")[-1] for f in target_files}
            if current_file_names != target_file_names:
                return True
            current_dir_names = {d.split("/")[-1] for d in current_dirs}
            target_dir_names = {d.split("/")[-1] for d in target_dirs}
            if current_dir_names != target_dir_names:
                return True
            for current_file in current_files:
                if self._file_change_status.get(current_file, True):
                    return True
            for current_dir in current_dirs:
                if self._dir_change_status.get(current_dir, True):
                    return True
            return False
        except Exception:
            return True

    async def _read_existing_overview_abstract(
        self, dir_uri: str
    ) -> tuple[Optional[str], Optional[str]]:
        target_path = self._get_target_file_path(dir_uri)
        if not target_path:
            return None, None
        try:
            overview = await self._viking_fs.read_file(f"{target_path}/.overview.md", ctx=self._ctx)
            abstract = await self._viking_fs.read_file(f"{target_path}/.abstract.md", ctx=self._ctx)
            return overview, abstract
        except Exception:
            return None, None

    async def _file_summary_task(self, parent_uri: str, file_path: str) -> None:
        """Generate file summary and notify parent completion."""

        file_name = file_path.split("/")[-1]
        need_vectorize = True
        # Perf instrumentation (Stage 1A): bracket the whole task so we can see
        # how long each file spends in the summary phase end-to-end. Pair this
        # with the per-step breakdown in semantic_processor._generate_text_summary.
        _t_task_start = time.monotonic()
        logger.info("[summary] phase=start file=%s", file_path)
        try:
            summary_dict = None
            if self._incremental_update:
                content_changed = await self._check_file_content_changed(file_path)
                self._file_change_status[file_path] = content_changed

                if not content_changed:
                    summary_dict = await self._read_existing_summary(file_path)
                    if summary_dict is not None:
                        need_vectorize = False
                    else:
                        self._file_change_status[file_path] = True
            else:
                self._file_change_status[file_path] = True
            if summary_dict is None:
                summary_dict = await self._processor._generate_single_file_summary(
                    file_path, llm_sem=self._llm_sem, ctx=self._ctx
                )
        except Exception as e:
            logger.warning(f"Failed to generate summary for {file_path}: {e}")
            summary_dict = {"name": file_name, "summary": ""}
        finally:
            self._stats.done_nodes += 1
            self._stats.in_progress_nodes = max(0, self._stats.in_progress_nodes - 1)
            logger.info(
                "[summary] phase=done file=%s wall_ms=%.1f need_vectorize=%s",
                file_path,
                (time.monotonic() - _t_task_start) * 1000,
                need_vectorize,
            )

        # Strip the internal `_prefetched_text` payload that the text summary
        # generator attaches so the embedding stage can skip a second AGFS
        # read. We pull it out *before* enqueueing the vectorize task and
        # *before* the dict is recorded in the parent node's file_summaries
        # (which is serialized into .overview.md downstream).
        prefetched_text = summary_dict.pop("_prefetched_text", None) if isinstance(
            summary_dict, dict
        ) else None

        try:
            if need_vectorize:
                use_summary = self._is_code_repo and bool(summary_dict.get("summary"))
                task = VectorizeTask(
                    task_type="file",
                    uri=file_path,
                    context_type=self._context_type,
                    ctx=self._ctx,
                    semantic_msg_id=self._semantic_msg_id,
                    file_path=file_path,
                    summary_dict=summary_dict,
                    parent_uri=parent_uri,
                    use_summary=use_summary,
                    prefetched_text=prefetched_text,
                )
                await self._add_vectorize_task(task)
        except Exception as e:
            logger.error(f"Failed to schedule vectorization for {file_path}: {e}", exc_info=True)
        await self._on_file_done(parent_uri, file_path, summary_dict)

    async def _on_file_done(
        self, parent_uri: str, file_path: str, summary_dict: Dict[str, str]
    ) -> None:
        node = self._nodes.get(parent_uri)
        if not node:
            return

        async with node.lock:
            idx = node.file_index.get(file_path)
            if idx is not None:
                node.file_summaries[idx] = summary_dict
            node.pending -= 1
            if node.pending == 0 and not node.overview_scheduled:
                node.overview_scheduled = True
                self._stats.pending_nodes = max(0, self._stats.pending_nodes - 1)
                self._stats.in_progress_nodes += 1
                asyncio.create_task(self._overview_task(parent_uri))

    async def _on_child_done(self, parent_uri: str, child_uri: str, abstract: str) -> None:
        node = self._nodes.get(parent_uri)
        if not node:
            return

        child_name = child_uri.split("/")[-1]
        async with node.lock:
            idx = node.child_index.get(child_uri)
            if idx is not None:
                node.children_abstracts[idx] = {"name": child_name, "abstract": abstract}
            node.pending -= 1
            if node.pending == 0 and not node.overview_scheduled:
                node.overview_scheduled = True
                self._stats.pending_nodes = max(0, self._stats.pending_nodes - 1)
                self._stats.in_progress_nodes += 1
                asyncio.create_task(self._overview_task(parent_uri))

    def _schedule_overview(self, dir_uri: str) -> None:
        node = self._nodes.get(dir_uri)
        if not node:
            return
        if node.overview_scheduled:
            return
        node.overview_scheduled = True
        self._stats.pending_nodes = max(0, self._stats.pending_nodes - 1)
        self._stats.in_progress_nodes += 1
        asyncio.create_task(self._overview_task(dir_uri))

    def _finalize_file_summaries(self, node: DirNode) -> List[Dict[str, str]]:
        summaries: List[Dict[str, str]] = []
        for idx, file_path in enumerate(node.file_paths):
            item = node.file_summaries[idx]
            if item is None:
                summaries.append({"name": file_path.split("/")[-1], "summary": ""})
            else:
                summaries.append(item)
        return summaries

    @property
    def stale(self) -> bool:
        return self._stale

    async def _finalize_children_abstracts(
        self, node: DirNode
    ) -> tuple[List[Dict[str, str]], int]:
        results: List[Dict[str, str]] = []
        abstract_misses = 0
        for idx, child_uri in enumerate(node.children_dirs):
            item = node.children_abstracts[idx]
            if item is None:
                abstract_misses += 1
                try:
                    abstract = await self._viking_fs.abstract(child_uri, ctx=self._ctx)
                except Exception:
                    abstract = ""
                results.append({"name": child_uri.split("/")[-1], "abstract": abstract})
            else:
                results.append(item)
        return results, abstract_misses

    def _is_stale(self) -> bool:
        from openviking.storage.queuefs.semantic_queue import is_semantic_coalesce_stale

        return is_semantic_coalesce_stale(self._coalesce_key, self._coalesce_version)

    async def _write_directory_semantics(
        self,
        dir_uri: str,
        overview: str,
        abstract: str,
        timing: Optional[Dict[str, float]] = None,
    ) -> bool:
        wrote = await write_semantic_sidecars(
            viking_fs=self._viking_fs,
            dir_uri=dir_uri,
            overview=overview,
            abstract=abstract,
            ctx=self._ctx,
            is_stale=self._is_stale,
            lock=self._lock,
            log_prefix="[SemanticDag]",
            timing=timing,
        )
        if not wrote:
            self._stale = True
        return wrote

    async def _overview_task(self, dir_uri: str) -> None:
        node = self._nodes.get(dir_uri)
        if not node:
            return
        need_vectorize = True
        children_changed = True
        abstract = ""
        # Perf instrumentation (no logic change): bracket each step of the
        # bottom-up directory roll-up — the silent gap after leaf summaries.
        _t_task = time.monotonic()
        if self._rollup_t0 is None:
            self._rollup_t0 = _t_task
        incr_ms = read_ms = children_ms = llm_ms = sidecar_ms = 0.0
        abstract_misses = 0
        branch = "generated"
        sidecar_timing: Dict[str, float] = {}
        try:
            overview = None
            abstract = None
            if self._incremental_update:
                _t = time.monotonic()
                children_changed = await self._check_dir_children_changed(
                    dir_uri, node.file_paths, node.children_dirs
                )
                incr_ms = (time.monotonic() - _t) * 1000

                if not children_changed:
                    need_vectorize = False
                    _t = time.monotonic()
                    overview, abstract = await self._read_existing_overview_abstract(dir_uri)
                    read_ms = (time.monotonic() - _t) * 1000
            if overview is None or abstract is None:
                _t = time.monotonic()
                async with node.lock:
                    file_summaries = self._finalize_file_summaries(node)
                    children_abstracts, abstract_misses = (
                        await self._finalize_children_abstracts(node)
                    )
                children_ms = (time.monotonic() - _t) * 1000
                _t = time.monotonic()
                async with self._llm_sem:
                    overview = await self._processor._generate_overview(
                        dir_uri, file_summaries, children_abstracts
                    )
                llm_ms = (time.monotonic() - _t) * 1000
                abstract = self._processor._extract_abstract_from_overview(overview)
                overview, abstract = self._processor._enforce_size_limits(overview, abstract)
            else:
                branch = "unchanged_reread"

            # Write directly, protected by the outer semantic lock. Bound the
            # number of directories writing at once (self._write_sem) so the
            # roll-up doesn't saturate S3 Express. sidecar_ms includes any time
            # spent queued on the write semaphore.
            _t = time.monotonic()
            try:
                async with self._write_sem:
                    wrote = await self._write_directory_semantics(
                        dir_uri, overview, abstract, timing=sidecar_timing
                    )
                if not wrote:
                    need_vectorize = False
            except Exception:
                logger.info(f"[SemanticDag] {dir_uri} write failed, skipping")
            sidecar_ms = (time.monotonic() - _t) * 1000

            try:
                if need_vectorize:
                    task = VectorizeTask(
                        task_type="directory",
                        uri=dir_uri,
                        context_type=self._context_type,
                        ctx=self._ctx,
                        semantic_msg_id=self._semantic_msg_id,
                        abstract=abstract,
                        overview=overview,
                    )
                    await self._add_vectorize_task(task)
            except Exception as e:
                logger.error(f"Failed to schedule vectorization for {dir_uri}: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Failed to generate overview for {dir_uri}: {e}", exc_info=True)
        finally:
            self._stats.done_nodes += 1
            self._stats.in_progress_nodes = max(0, self._stats.in_progress_nodes - 1)
            total_ms = (time.monotonic() - _t_task) * 1000
            lock_acquire_ms = sidecar_timing.get("lock_acquire_ms", 0.0)
            write_only_ms = sidecar_timing.get("write_ms", 0.0)
            rel = dir_uri[len(self._root_uri):].strip("/") if self._root_uri else ""
            depth = 0 if not rel else rel.count("/") + 1
            self._overview_acc["dirs"] += 1
            self._overview_acc["lock"] += lock_acquire_ms
            self._overview_acc["llm"] += llm_ms
            self._overview_acc["read_existing"] += read_ms
            self._overview_acc["children"] += children_ms
            self._overview_acc["write"] += write_only_ms
            if branch == "unchanged_reread":
                self._overview_acc["unchanged"] += 1
            else:
                self._overview_acc["generated"] += 1
            logger.info(
                "[overview] dir=%s branch=%s depth=%d total_ms=%.1f incr_ms=%.1f "
                "read_ms=%.1f children_ms=%.1f abstract_misses=%d llm_ms=%.1f "
                "sidecar_ms=%.1f lock_acquire_ms=%.1f write_ms=%.1f",
                dir_uri, branch, depth, total_ms, incr_ms, read_ms, children_ms,
                abstract_misses, llm_ms, sidecar_ms, lock_acquire_ms, write_only_ms,
            )

        self._dir_change_status[dir_uri] = children_changed

        parent_uri = self._parent.get(dir_uri)
        if parent_uri is None:
            acc = self._overview_acc
            rollup_ms = (
                (time.monotonic() - self._rollup_t0) * 1000 if self._rollup_t0 else 0.0
            )
            logger.info(
                "[overview_rollup] total_ms=%.1f dirs=%d ops=[lock=%.1f llm=%.1f "
                "read_existing=%.1f children=%.1f write=%.1f] "
                "branches=[unchanged=%d generated=%d]",
                rollup_ms, acc["dirs"], acc["lock"], acc["llm"], acc["read_existing"],
                acc["children"], acc["write"], acc["unchanged"], acc["generated"],
            )
            if self._root_done:
                self._root_done.set()
            return

        await self._on_child_done(parent_uri, dir_uri, abstract)

    async def _add_vectorize_task(self, task: VectorizeTask) -> None:
        """Add a vectorize task to the pending list."""
        if self._skip_vectorization:
            logger.info(
                "Skipping vectorization task for %s (requested via SemanticMsg)",
                task.uri,
            )
            return
        async with self._vectorize_lock:
            self._pending_vectorize_tasks.append(task)
            if task.task_type == "file":
                self._vectorize_task_count += 1
            else:  # directory
                self._vectorize_task_count += 2

    def get_stats(self) -> DagStats:
        return DagStats(
            total_nodes=self._stats.total_nodes,
            pending_nodes=self._stats.pending_nodes,
            in_progress_nodes=self._stats.in_progress_nodes,
            done_nodes=self._stats.done_nodes,
        )


if False:  # pragma: no cover - for type checkers only
    from openviking.storage.queuefs.semantic_processor import SemanticProcessor

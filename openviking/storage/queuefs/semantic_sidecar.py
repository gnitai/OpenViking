# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Shared writeback for semantic sidecar files."""

import asyncio
import time
from typing import Any, Callable, Dict, Optional

from openviking.server.identity import RequestContext
from openviking.storage.transaction import NO_LOCK, LockLease
from openviking_cli.utils.logger import get_logger

logger = get_logger(__name__)


async def write_semantic_sidecars(
    *,
    viking_fs: Any,
    dir_uri: str,
    overview: str,
    abstract: str,
    ctx: Optional[RequestContext],
    is_stale: Callable[[], bool],
    lock: LockLease = NO_LOCK,
    log_prefix: str = "[Semantic]",
    timing: Optional[Dict[str, float]] = None,
) -> bool:
    if is_stale():
        logger.info("%s Skipping stale semantic write for %s", log_prefix, dir_uri)
        return False

    try:
        from openviking.storage.transaction import LockContext, get_lock_manager

        lock_manager = get_lock_manager()
    except Exception:
        await _write_sidecars(viking_fs, dir_uri, overview, abstract, ctx)
        return True

    lock_paths = [
        viking_fs._uri_to_path(f"{dir_uri}/.overview.md", ctx=ctx),
        viking_fs._uri_to_path(f"{dir_uri}/.abstract.md", ctx=ctx),
    ]
    # Perf instrumentation: split the per-directory sidecar cost into lock-acquire
    # vs the two write_file calls. Manual enter/exit (instead of `async with`) so
    # we can time __aenter__ on its own; __aexit__ only releases and never
    # suppresses, so the finally is safe.
    ctx_mgr = LockContext(lock_manager, lock_paths, lock_mode="exact", handle=lock.handle)
    _t_lock = time.monotonic()
    await ctx_mgr.__aenter__()
    lock_acquire_ms = (time.monotonic() - _t_lock) * 1000
    write_ms = 0.0
    try:
        if is_stale():
            logger.info("%s Skipping stale semantic write for %s", log_prefix, dir_uri)
            return False
        _t_write = time.monotonic()
        await _write_sidecars(viking_fs, dir_uri, overview, abstract, ctx)
        write_ms = (time.monotonic() - _t_write) * 1000
        return True
    finally:
        await ctx_mgr.__aexit__(None, None, None)
        if timing is not None:
            timing["lock_acquire_ms"] = lock_acquire_ms
            timing["write_ms"] = write_ms
        logger.info(
            "%s [sidecar] dir=%s lock_acquire_ms=%.1f write_ms=%.1f",
            log_prefix,
            dir_uri,
            lock_acquire_ms,
            write_ms,
        )


async def _write_sidecars(
    viking_fs: Any,
    dir_uri: str,
    overview: str,
    abstract: str,
    ctx: Optional[RequestContext],
) -> None:
    # The two sidecars are independent objects; write them concurrently so the
    # per-write S3 latency overlaps instead of stacking.
    await asyncio.gather(
        viking_fs.write_file(f"{dir_uri}/.overview.md", overview, ctx=ctx),
        viking_fs.write_file(f"{dir_uri}/.abstract.md", abstract, ctx=ctx),
    )

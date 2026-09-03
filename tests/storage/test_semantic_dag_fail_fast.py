# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""An auth-class LLM error (401/403) must abort the semantic DAG instead of
writing empty summaries for every file and then vectorizing the whole tree.

In prod a bad LLM-gateway token made every one of a 676-file repo's summaries
401, the run still "completed" with empty semantics and 676 embeddings, and
because nothing usable was cached the next write repeated all of it."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.storage.queuefs.semantic_dag import SemanticDagExecutor, SemanticRunAborted
from openviking_cli.session.user_id import UserIdentifier

ROOT = "wfs://resources/code/repo/acme-main"
TREE = {
    ROOT: [
        {"name": "a.py", "isDir": False},
        {"name": "b.py", "isDir": False},
        {"name": "sub", "isDir": True},
    ],
    f"{ROOT}/sub": [{"name": "c.py", "isDir": False}],
}


def _mock_transaction_layer(monkeypatch):
    mock_handle = MagicMock()
    monkeypatch.setattr(
        "openviking.storage.transaction.lock_context.LockContext.__aenter__",
        AsyncMock(return_value=mock_handle),
    )
    monkeypatch.setattr(
        "openviking.storage.transaction.lock_context.LockContext.__aexit__",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        "openviking.storage.transaction.get_lock_manager",
        lambda: MagicMock(),
    )


class _FakeVikingFS:
    def __init__(self, tree):
        self._tree = tree
        self.writes = []

    async def ls(self, uri, ctx=None):
        return self._tree.get(uri, [])

    async def write_file(self, path, content, ctx=None):
        self.writes.append((path, content))

    def _uri_to_path(self, uri, ctx=None):
        return uri.replace("wfs://", "/local/acc1/")


class _FailingProcessor:
    def __init__(self, error: Exception):
        self._error = error
        self.summary_calls = 0
        self.overview_calls = 0
        self.vectorized = []

    async def _generate_single_file_summary(self, file_path, llm_sem=None, ctx=None):
        self.summary_calls += 1
        raise self._error

    async def _generate_overview(self, dir_uri, file_summaries, children_abstracts):
        self.overview_calls += 1
        return "overview"

    def _extract_abstract_from_overview(self, overview):
        return "abstract"

    def _enforce_size_limits(self, overview, abstract):
        return overview, abstract

    async def _vectorize_directory(
        self, uri, context_type, abstract, overview, ctx=None, semantic_msg_id=None
    ):
        self.vectorized.append(uri)

    async def _vectorize_single_file(
        self,
        parent_uri,
        context_type,
        file_path,
        summary_dict,
        ctx=None,
        semantic_msg_id=None,
        use_summary=False,
        prefetched_text=None,
    ):
        self.vectorized.append(file_path)


class _RecordingTracker:
    def __init__(self):
        self.registered = []

    async def register(self, **kwargs):
        self.registered.append(kwargs)


def _executor(monkeypatch, processor):
    _mock_transaction_layer(monkeypatch)
    fake_fs = _FakeVikingFS(TREE)
    tracker = _RecordingTracker()
    monkeypatch.setattr("openviking.storage.queuefs.semantic_dag.get_viking_fs", lambda: fake_fs)
    monkeypatch.setattr(
        "openviking.storage.queuefs.embedding_tracker.EmbeddingTaskTracker.get_instance",
        lambda: tracker,
    )
    ctx = RequestContext(user=UserIdentifier("acc1", "user1", "agent1"), role=Role.USER)
    executor = SemanticDagExecutor(
        processor=processor,
        context_type="resource",
        max_concurrent_llm=1,
        ctx=ctx,
    )
    return executor, fake_fs, tracker


@pytest.mark.asyncio
async def test_permanent_llm_error_aborts_run_without_writing_or_vectorizing(monkeypatch):
    error = RuntimeError(
        "Error code: 401 - {'code': 'INVALID_BEARER_TOKEN', 'message': 'invalid_bearer_token'}"
    )
    processor = _FailingProcessor(error)
    executor, fake_fs, tracker = _executor(monkeypatch, processor)

    with pytest.raises(SemanticRunAborted, match="401") as excinfo:
        await executor.run(ROOT)
    assert excinfo.value.cause is error

    # First failure aborts the run: no further LLM calls, nothing written or embedded.
    assert processor.summary_calls == 1
    assert processor.overview_calls == 0
    assert fake_fs.writes == []
    assert tracker.registered == []
    assert processor.vectorized == []


@pytest.mark.asyncio
async def test_transient_llm_error_still_degrades_to_empty_summary(monkeypatch):
    processor = _FailingProcessor(RuntimeError("Error code: 503 - upstream unavailable"))
    executor, fake_fs, tracker = _executor(monkeypatch, processor)

    await executor.run(ROOT)

    assert processor.summary_calls == 3
    assert processor.overview_calls == 2
    assert len(tracker.registered) == 1


@pytest.mark.asyncio
async def test_non_auth_permanent_error_still_degrades_per_file(monkeypatch):
    """A 400 is specific to one file's content; it must not fail the whole run."""
    processor = _FailingProcessor(RuntimeError("Error code: 400 - bad request"))
    executor, fake_fs, tracker = _executor(monkeypatch, processor)

    await executor.run(ROOT)

    assert processor.summary_calls == 3
    assert processor.overview_calls == 2
    assert len(tracker.registered) == 1

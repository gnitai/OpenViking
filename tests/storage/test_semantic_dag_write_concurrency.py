# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""The directory overview roll-up must bound how many directories write their
sidecars to storage at once. Without a cap, all sibling dirs flush their
.overview.md/.abstract.md concurrently and saturate S3 Express (throttling).
`write_concurrency` caps the number of directories writing simultaneously."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.storage.queuefs.semantic_dag import SemanticDagExecutor
from openviking_cli.session.user_id import UserIdentifier


class _PeakFS:
    def __init__(self, tree):
        self._tree = tree
        self.inflight = 0
        self.peak = 0

    async def ls(self, uri, ctx=None):
        return self._tree.get(uri, [])

    async def write_file(self, path, content, ctx=None):
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        await asyncio.sleep(0.01)  # hold the slot so overlap is observable
        self.inflight -= 1

    def _uri_to_path(self, uri, ctx=None):
        return uri.replace("viking://", "/local/acc1/")


class _FakeProcessor:
    async def _generate_single_file_summary(self, file_path, llm_sem=None, ctx=None):
        return {"name": file_path.split("/")[-1], "summary": "s"}

    async def _generate_overview(self, dir_uri, file_summaries, children_abstracts):
        return "overview"

    def _extract_abstract_from_overview(self, overview):
        return "abstract"

    def _enforce_size_limits(self, overview, abstract):
        return overview, abstract

    async def _vectorize_directory(self, *a, **k):
        pass

    async def _vectorize_single_file(self, *a, **k):
        pass


class _DummyTracker:
    async def register(self, **_kwargs):
        return None


def _patch(monkeypatch, fake_fs):
    monkeypatch.setattr("openviking.storage.queuefs.semantic_dag.get_viking_fs", lambda: fake_fs)
    monkeypatch.setattr(
        "openviking.storage.queuefs.embedding_tracker.EmbeddingTaskTracker.get_instance",
        lambda: _DummyTracker(),
    )
    monkeypatch.setattr(
        "openviking.storage.transaction.lock_context.LockContext.__aenter__",
        AsyncMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        "openviking.storage.transaction.lock_context.LockContext.__aexit__",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        "openviking.storage.transaction.get_lock_manager", lambda: MagicMock()
    )


@pytest.mark.asyncio
async def test_write_concurrency_caps_simultaneous_sidecar_writes(monkeypatch):
    root_uri = "viking://resources/root"
    # 4 sibling directories whose overview tasks would otherwise flush together.
    tree = {root_uri: [{"name": f"child{i}", "isDir": True} for i in range(4)]}
    for i in range(4):
        tree[f"{root_uri}/child{i}"] = [{"name": f"f{i}.txt", "isDir": False}]

    fake_fs = _PeakFS(tree)
    _patch(monkeypatch, fake_fs)

    ctx = RequestContext(user=UserIdentifier("acc1", "user1", "agent1"), role=Role.USER)
    executor = SemanticDagExecutor(
        processor=_FakeProcessor(),
        context_type="resource",
        max_concurrent_llm=8,
        ctx=ctx,
        skip_vectorization=True,
        write_concurrency=1,
    )
    await executor.run(root_uri)
    await asyncio.sleep(0)

    # With write_concurrency=1 only one directory writes at a time; the two
    # sidecars within that directory go in parallel (gather) → peak 2.
    assert fake_fs.peak <= 2, (
        f"peak concurrent sidecar writes was {fake_fs.peak}; expected <= 2 "
        "with write_concurrency=1 (write fan-out is not bounded)"
    )
    assert fake_fs.peak >= 1


if __name__ == "__main__":
    pytest.main([__file__])

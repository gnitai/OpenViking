# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""An aborted semantic run (auth-class LLM error) fails only its own request:
it must not trip the processor-wide circuit breaker (which would pause every
project's semantic processing for the reset window) and must not be re-enqueued."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.storage.queuefs.semantic_dag import SemanticRunAborted
from openviking.storage.queuefs.semantic_msg import SemanticMsg
from openviking.storage.queuefs.semantic_processor import SemanticProcessor
from openviking.storage.transaction import NO_LOCK
from openviking.telemetry.request_wait_tracker import get_request_wait_tracker

ROOT = "wfs://resources/code/repo/acme-main/src"
TELEMETRY_ID = "tm_abort_test"


class _FakeVikingFS:
    async def exists(self, uri, ctx=None):
        return True


class _AbortingDagExecutor:
    def __init__(self, **kwargs):
        self.stale = False

    async def run(self, root_uri):
        raise SemanticRunAborted(RuntimeError("Error code: 401 - {'code': 'INVALID_BEARER_TOKEN'}"))

    def get_stats(self):
        from openviking.storage.queuefs.semantic_dag import DagStats

        return DagStats()


@pytest.mark.asyncio
async def test_aborted_run_fails_request_without_tripping_breaker(monkeypatch):
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.get_viking_fs",
        lambda: _FakeVikingFS(),
    )
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.SemanticDagExecutor",
        _AbortingDagExecutor,
    )
    monkeypatch.setattr(
        "openviking.storage.queuefs.semantic_processor.SemanticLockScope.resolve",
        AsyncMock(return_value=SimpleNamespace(lock=NO_LOCK, close=AsyncMock())),
    )
    processor = SemanticProcessor()
    processor._enqueue_parent_refresh = AsyncMock()
    processor._reenqueue_semantic_msg = AsyncMock()

    tracker = get_request_wait_tracker()
    tracker.register_request(TELEMETRY_ID)
    msg = SemanticMsg(
        uri=ROOT,
        context_type="resource",
        telemetry_id=TELEMETRY_ID,
        changes={"modified": [f"{ROOT}/App.jsx"]},
    )
    tracker.register_semantic_root(TELEMETRY_ID, msg.id)
    try:
        await processor.on_dequeue(msg.to_dict())

        # The request is terminal and failed with the auth error...
        assert tracker.is_complete(TELEMETRY_ID)
        status = tracker.build_queue_status(TELEMETRY_ID)
        assert status["Semantic"]["error_count"] == 1
        assert "401" in status["Semantic"]["errors"][0]["message"]
        # ...but nothing was re-enqueued and the shared breaker stayed closed.
        processor._reenqueue_semantic_msg.assert_not_called()
        processor._circuit_breaker.check()
    finally:
        tracker.cleanup(TELEMETRY_ID)

# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Tests for retrieval usage instrumentation emitted by SearchService."""

from __future__ import annotations

from types import SimpleNamespace

from openviking.observability.events import (
    ObservabilityEvent,
    register_event_subscriber,
    reset_event_bus_for_tests,
    unregister_event_subscriber,
)
from openviking.observability.usage_audit.projection import project_events
from openviking.service.search_service import _emit_retrieval_call


def _fake_ctx(account_id: str, user_id: str, agent_id: str) -> SimpleNamespace:
    # Mirrors the attributes _emit_retrieval_call reads from RequestContext:
    # ctx.account_id, ctx.user.user_id, ctx.user.agent_id.
    return SimpleNamespace(
        account_id=account_id,
        user=SimpleNamespace(user_id=user_id, agent_id=agent_id),
    )


def _fake_result() -> SimpleNamespace:
    memory = SimpleNamespace(abstract="hello world", overview="a longer overview text")
    return SimpleNamespace(memories=[memory], resources=[], skills=[], total=1)


def test_emit_retrieval_call_attributes_identity_from_ctx():
    """account_id must come from ctx, not ambient observability context."""
    captured: list[ObservabilityEvent] = []
    reset_event_bus_for_tests()
    register_event_subscriber("test-capture", captured.append)
    try:
        # No observability root context is set here, so if attribution relied on
        # ambient context the event would bucket under "__unknown__".
        _emit_retrieval_call("find", _fake_result(), _fake_ctx("acct-9", "user-9", "agent-9"))
    finally:
        unregister_event_subscriber("test-capture")

    assert len(captured) == 1
    event = captured[0]
    assert event.event_name == "retrieval.call"
    assert event.account_id == "acct-9"
    assert event.payload["operation"] == "find"
    assert event.payload["result_count"] == 1
    assert event.payload["result_token_count"] > 0

    # And it projects into the correct per-account retrieval bucket.
    projection = project_events([event])
    keys = list(projection.retrieval_rows.keys())
    assert len(keys) == 1
    account_id = keys[0][0]
    operation = keys[0][5]
    assert account_id == "acct-9"
    assert operation == "find"
    _, results, tokens = projection.retrieval_rows[keys[0]]
    assert results == 1
    assert tokens == event.payload["result_token_count"]

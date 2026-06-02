# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Per-process, per-project lazy initialization guard."""

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.server.project_init import ProjectInitGuard, build_default_initializer
from openviking_cli.session.user_id import UserIdentifier


def _ctx(account: str, user: str) -> RequestContext:
    return RequestContext(
        user=UserIdentifier(account, user, "default"),
        role=Role.ROOT,
    )


@pytest.mark.asyncio
async def test_runs_initializer_once_per_account_user():
    calls = []

    async def _init(ctx):
        calls.append((ctx.account_id, ctx.user.user_id))

    guard = ProjectInitGuard(initializer=_init)

    await guard.ensure(_ctx("3493", "m1"))
    await guard.ensure(_ctx("3493", "m1"))  # second time is a no-op

    assert calls == [("3493", "m1")]


@pytest.mark.asyncio
async def test_distinct_keys_each_run_once():
    calls = []

    async def _init(ctx):
        calls.append((ctx.account_id, ctx.user.user_id))

    guard = ProjectInitGuard(initializer=_init)

    await guard.ensure(_ctx("3493", "m1"))
    await guard.ensure(_ctx("3493", "m2"))  # same account, different user
    await guard.ensure(_ctx("999", "m1"))  # different account

    assert calls == [("3493", "m1"), ("3493", "m2"), ("999", "m1")]


@pytest.mark.asyncio
async def test_failed_init_is_retried_and_not_marked_done():
    calls = []

    async def _init(ctx):
        calls.append((ctx.account_id, ctx.user.user_id))
        if len(calls) == 1:
            raise RuntimeError("boom")

    guard = ProjectInitGuard(initializer=_init)

    with pytest.raises(RuntimeError, match="boom"):
        await guard.ensure(_ctx("3493", "m1"))

    # The failure must NOT mark the key done; the next ensure retries.
    await guard.ensure(_ctx("3493", "m1"))
    assert calls == [("3493", "m1"), ("3493", "m1")]

    # Now it succeeded; a third ensure is a no-op.
    await guard.ensure(_ctx("3493", "m1"))
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_default_is_preseeded():
    async def _init(ctx):  # pragma: no cover - must not run for preseeded key
        raise AssertionError("initializer should not run for the preseeded key")

    guard = ProjectInitGuard(initializer=_init)

    # The startup path already initializes the default account; the guard
    # pre-seeds the default key so a default ctx that reaches it is never
    # re-initialized. Auth coerces both account and user to "default", so the
    # key is "default::default".
    default_ctx = _ctx("default", "default")
    await guard.ensure(default_ctx)  # must NOT invoke the initializer


@pytest.mark.asyncio
async def test_default_initializer_applies_schema_before_directory_init():
    """ensure_collection MUST run before directory init.

    Directory init auto-generates L0/L1 sidecars whose embeddings are written
    to the project's vector namespace. If the schema-bearing collection has not
    been applied to the per-project adapter yet, those early writes fail
    ("Collection ... does not exist") and the preset-dir / memory-scaffold
    sidecars are lost. Pin the order so a refactor cannot silently reintroduce
    the bug.
    """
    order = []

    class _FakeVikingDBManager:
        async def ensure_collection(self, *, ctx):
            order.append("ensure_collection")
            return True

    class _FakeService:
        def __init__(self):
            self.vikingdb_manager = _FakeVikingDBManager()

        async def initialize_account_directories(self, ctx):
            order.append("initialize_account_directories")
            return 0

        async def initialize_user_directories(self, ctx):
            order.append("initialize_user_directories")
            return 0

    initializer = build_default_initializer(_FakeService())
    await initializer(_ctx("3493", "m1"))

    assert order[0] == "ensure_collection", (
        f"ensure_collection must run first; got order {order}"
    )
    assert order == [
        "ensure_collection",
        "initialize_account_directories",
        "initialize_user_directories",
    ]

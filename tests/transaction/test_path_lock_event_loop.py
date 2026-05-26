# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Regression test for the S3 ingestion event-loop starvation bug.

Before the lock-manager async fix, PathLockEngine called the sync AGFSClient
directly from `async def` methods. On the S3 backend each call is a 100-500ms
HTTPS round trip, so the event loop was blocked for the full duration of an
`acquire_tree` -- starving httpx (OpenAI) and APScheduler. This test guards
against regressions by simulating slow AGFS and asserting a short concurrent
`asyncio.sleep` task still runs on time while `acquire_tree` is in flight.
"""

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from openviking.storage.transaction.lock_handle import LockHandle
from openviking.storage.transaction.path_lock import PathLockEngine


def _slow_agfs(latency_s: float = 0.2) -> MagicMock:
    """Build a sync AGFS mock that sleeps `latency_s` on every call.

    Mirrors the worst-case S3 round-trip cost. Returns reasonable defaults
    (no existing lock, empty dirs, successful writes) so `acquire_tree`
    completes after one full path-lock acquisition cycle.
    """

    agfs = MagicMock()

    def _sleep(*_args, **_kwargs):
        time.sleep(latency_s)

    def _read(*_args, **_kwargs):
        time.sleep(latency_s)
        raise Exception("not found")

    def _stat(*_args, **_kwargs):
        time.sleep(latency_s)
        return {"isDir": True}

    def _ls(*_args, **_kwargs):
        time.sleep(latency_s)
        return []

    def _write(*_args, **_kwargs):
        time.sleep(latency_s)
        return None

    def _rm(*_args, **_kwargs):
        time.sleep(latency_s)
        return None

    def _mkdir(*_args, **_kwargs):
        time.sleep(latency_s)
        return None

    agfs.read.side_effect = _read
    agfs.stat.side_effect = _stat
    agfs.ls.side_effect = _ls
    agfs.write.side_effect = _write
    agfs.rm.side_effect = _rm
    agfs.mkdir.side_effect = _mkdir
    return agfs


@pytest.mark.asyncio
async def test_acquire_tree_does_not_block_event_loop() -> None:
    """`acquire_tree` must keep the event loop free even when AGFS is slow.

    With a 200ms artificial latency per AGFS call, a concurrent
    `asyncio.sleep(0.01)` must still fire within a small bounded delay
    (we allow 200ms slack to accommodate CI jitter and the cost of moving
    work to the AGFS thread pool).
    """

    engine = PathLockEngine(_slow_agfs(0.2))
    owner = LockHandle(id="test-owner")

    sleep_started = asyncio.get_running_loop().time()
    sleep_done: dict[str, float] = {}

    async def _short_sleep() -> None:
        await asyncio.sleep(0.01)
        sleep_done["t"] = asyncio.get_running_loop().time() - sleep_started

    acquire_task = asyncio.create_task(
        engine.acquire_tree("/local/test/event-loop", owner, timeout=10.0)
    )
    short_task = asyncio.create_task(_short_sleep())

    await short_task
    assert "t" in sleep_done, "short sleep never ran"
    # The event loop should remain responsive: the 10ms sleep should complete
    # well before any single AGFS call (200ms). 200ms is a generous bound
    # that still catches the original bug (which would block for >1s).
    assert sleep_done["t"] < 0.2, (
        f"concurrent sleep was delayed {sleep_done['t']:.3f}s -- "
        "looks like acquire_tree is blocking the event loop again"
    )

    await acquire_task

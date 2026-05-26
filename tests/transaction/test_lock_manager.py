# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Tests for LockManager."""

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from openviking.storage.transaction.lock_manager import LockManager
from openviking.storage.transaction.path_lock import LOCK_FILE_NAME


def _lock_file_gone(agfs_client, lock_path: str) -> bool:
    try:
        agfs_client.stat(lock_path)
        return False
    except Exception:
        return True


@pytest.fixture
def lm(agfs_client):
    return LockManager(agfs=agfs_client, lock_timeout=1.0, lock_expire=1.0)


class TestLockManagerBasic:
    async def test_create_handle_and_acquire_exact_path(self, agfs_client, lm, test_dir):
        handle = lm.create_handle()
        ok = await lm.acquire_exact_path(handle, test_dir)
        assert ok is True

        lock_path = handle.locks[0]
        content = agfs_client.cat(lock_path)
        assert content is not None

        await lm.release(handle)
        assert _lock_file_gone(agfs_client, lock_path)

    async def test_acquire_tree(self, agfs_client, lm, test_dir):
        handle = lm.create_handle()
        ok = await lm.acquire_tree(handle, test_dir)
        assert ok is True

        token = agfs_client.cat(f"{test_dir}/{LOCK_FILE_NAME}")
        token_str = token.decode("utf-8") if isinstance(token, bytes) else token
        assert ":T" in token_str

        await lm.release(handle)

    async def test_acquire_mv(self, agfs_client, lm, test_dir):
        src = f"{test_dir}/mv-src-{uuid.uuid4().hex}"
        dst = f"{test_dir}/mv-dst-{uuid.uuid4().hex}"
        agfs_client.mkdir(src)
        agfs_client.mkdir(dst)

        handle = lm.create_handle()
        ok = await lm.acquire_mv(handle, src, dst)
        assert ok is True
        assert len(handle.locks) == 2

        await lm.release(handle)
        assert handle.id not in lm.get_active_handles()

    async def test_release_removes_from_active(self, lm, test_dir):
        handle = lm.create_handle()

        await lm.acquire_exact_path(handle, test_dir)
        assert handle.id in lm.get_active_handles()

        await lm.release(handle)

        assert handle.id not in lm.get_active_handles()

    async def test_stop_releases_all(self, agfs_client, lm, test_dir):
        h1 = lm.create_handle()
        h2 = lm.create_handle()
        await lm.acquire_exact_path(h1, test_dir)

        sub = f"{test_dir}/sub-{uuid.uuid4().hex}"
        agfs_client.mkdir(sub)
        await lm.acquire_exact_path(h2, sub)

        await lm.stop()
        assert len(lm.get_active_handles()) == 0

    async def test_exact_path_allows_missing_target(self, lm):
        handle = lm.create_handle()
        ok = await lm.acquire_exact_path(handle, "/local/nonexistent-xyz")
        assert ok is True

        await lm.release(handle)

    async def test_explicit_none_timeout_passes_through_as_infinite_wait(self, lm):
        handle = lm.create_handle()
        lm._path_lock = MagicMock()
        lm._path_lock.acquire_tree = AsyncMock(return_value=True)

        ok = await lm.acquire_tree(handle, "/local/test", timeout=None)

        assert ok is True
        lm._path_lock.acquire_tree.assert_awaited_once_with("/local/test", handle, timeout=None)

    async def test_recover_pending_redo_preserves_cancelled_error(self, lm):
        lm._redo_log = MagicMock()
        lm._redo_log.list_pending_async = AsyncMock(return_value=["redo-task"])
        lm._redo_log.read_async = AsyncMock(
            return_value={"archive_uri": "a", "session_uri": "b"}
        )
        lm._redo_log.mark_done_async = AsyncMock()
        lm._redo_session_memory = AsyncMock(side_effect=asyncio.CancelledError("shutdown"))

        with pytest.raises(asyncio.CancelledError):
            await lm._recover_pending_redo()

        lm._redo_log.mark_done_async.assert_not_awaited()

    async def test_start_skips_redo_recovery_when_disabled(self, client):
        lm_disabled = LockManager(
            agfs=client._client.service._agfs_client,
            lock_timeout=1.0,
            lock_expire=1.0,
            redo_recovery_enabled=False,
        )
        lm_disabled._recover_pending_redo = AsyncMock()

        await lm_disabled.start()
        await asyncio.sleep(0)

        assert lm_disabled._redo_task is None
        lm_disabled._recover_pending_redo.assert_not_called()

        await lm_disabled.stop()


class TestLockTypeCache:
    """Fix A (Stage 3): in-memory lock_type cache on LockHandle eliminates
    per-mv S3 verification reads. The race that fired at max_concurrent=64 — a
    concurrent lease-refresh write overlapping a consumer's ownership read —
    is gone because the consumer reads the cache instead of S3."""

    async def test_acquire_tree_populates_lock_type_cache(self, lm, test_dir):
        handle = lm.create_handle()
        await lm.acquire_tree(handle, test_dir)
        lock_path = handle.locks[0]
        assert handle.lock_types[lock_path] == "T"
        await lm.release(handle)
        assert lock_path not in handle.lock_types

    async def test_acquire_exact_path_populates_lock_type_cache(self, lm, test_dir):
        handle = lm.create_handle()
        await lm.acquire_exact_path(handle, test_dir)
        lock_path = handle.locks[0]
        assert handle.lock_types[lock_path] == "E"
        await lm.release(handle)
        assert lock_path not in handle.lock_types

    async def test_adopt_handle_populates_lock_type_cache(self, agfs_client, lm, test_dir):
        original = lm.create_handle()
        await lm.acquire_tree(original, test_dir)
        lock_path = original.locks[0]
        original_id = original.id

        lm._handles.pop(original_id, None)
        adopted = lm.adopt_handle(original_id, [lock_path])
        assert adopted is not None
        assert adopted.id == original_id
        assert lock_path in adopted.locks
        assert adopted.lock_types[lock_path] == "T"

        await lm.release(adopted)

    async def test_reusing_exact_lock_on_tree_path_does_not_downgrade_cache(
        self, agfs_client, lm, test_dir
    ):
        """Regression for Stage 3 Fix A bug. If a handle adopts a TREE lock at
        path P (via handoff), a later `acquire_exact_path(P)` short-circuits via
        the reuse branch but used to overwrite the cached lock_type from 'T' →
        'E'. Subsequent `_has_owned_ancestor_tree` checks for descendants would
        then see 'E' (not TREE) and time out waiting for an ancestor lock —
        exactly the failure that fired at max_concurrent=64 during SyncDiff."""
        parent_handle = lm.create_handle()
        await lm.acquire_tree(parent_handle, test_dir)
        lock_path = parent_handle.locks[0]
        assert parent_handle.lock_types[lock_path] == "T"

        # acquire_exact_path on the same dir path hits the reuse branch.
        ok = await lm.acquire_exact_path(parent_handle, test_dir)
        assert ok is True
        # The cached type must still be TREE — exact on the same path does not
        # weaken the ancestor-tree coverage.
        assert parent_handle.lock_types[lock_path] == "T", (
            "exact-on-tree reuse must not downgrade the cached lock_type"
        )

        # Descendant ancestor-walk still recognises the TREE lock from cache.
        child = f"{test_dir}/desc-{uuid.uuid4().hex}"
        assert await lm._path_lock._has_owned_ancestor_tree(child, parent_handle) is True

        await lm.release(parent_handle)

    async def test_ancestor_tree_check_uses_cache_no_s3_read(self, lm, test_dir):
        """The race-prone S3 read on every `_has_owned_ancestor_tree` call must
        be skipped when the cache is populated. Spy on `_read_token` to confirm
        no read happens during the ancestor walk."""
        parent_handle = lm.create_handle()
        await lm.acquire_tree(parent_handle, test_dir)

        read_count = {"n": 0}
        real_read = lm._path_lock._read_token

        def counting_read(lock_path: str):
            read_count["n"] += 1
            return real_read(lock_path)

        lm._path_lock._read_token = counting_read

        child = f"{test_dir}/child-{uuid.uuid4().hex}"
        before = read_count["n"]
        assert await lm._path_lock._has_owned_ancestor_tree(child, parent_handle) is True
        assert read_count["n"] == before, (
            "ancestor-tree check should hit the in-memory cache, not S3"
        )

        await lm.release(parent_handle)

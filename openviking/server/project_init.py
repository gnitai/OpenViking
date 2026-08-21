# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Lazy per-project initialization for dynamically-asserted projects.

A project IS an OpenViking ``account``. Projects appear dynamically: a
request with the ROOT api key + ``X-OpenViking-Account: <project_id>`` (plus
``X-OpenViking-User``) asserts an account that was never provisioned via the
admin API. On FIRST contact with such a project per process we must idempotently
(a) create the preset account/user directories and (b) apply the FULL,
schema-bearing ``create_collection`` to the project's own vector namespace
(``{collection}-{project_id}``). The schema must be (re)applied every process
because a fresh process builds a fresh adapter whose in-memory wrap has empty
``Fields`` — and Turbopuffer strips ``vector`` (and every attribute) from writes
while ``Fields`` is empty.

``ProjectInitGuard`` runs the initializer once per (account_id, user_id) per
process, success-gated (a failed init is NOT marked done, so it is retried) and
concurrency-safe (an asyncio lock with double-checked locking).
"""

import asyncio
from typing import Awaitable, Callable

from openviking.server.identity import RequestContext
from openviking.storage.project_embedding import ensure_project_pinned

# Initializer signature: takes a RequestContext, performs the bootstrap.
ProjectInitializer = Callable[[RequestContext], Awaitable[None]]


class ProjectInitGuard:
    """Run a per-project initializer once per (account_id, user_id) per process."""

    def __init__(self, initializer: ProjectInitializer):
        self._initializer = initializer
        # The default account is initialized at startup; pre-seed its sentinel
        # so a default ctx that reaches the guard is never re-initialized.
        self._done: set[str] = {"default::default"}
        self._lock = asyncio.Lock()

    @staticmethod
    def _key(ctx: RequestContext) -> str:
        return f"{ctx.account_id}::{ctx.user.user_id}"

    async def ensure(self, ctx: RequestContext) -> None:
        """Idempotently initialize the project for ``ctx`` (once per key)."""
        key = self._key(ctx)
        if key in self._done:
            return
        async with self._lock:
            # Double-checked: another coroutine may have completed init while we
            # awaited the lock.
            if key in self._done:
                return
            await self._initializer(ctx)
            # Only mark done AFTER a successful init, so a failed init is retried.
            self._done.add(key)


def build_default_initializer(service) -> ProjectInitializer:
    """Build the standard initializer: preset dirs + schema-applied namespace."""

    async def _init(ctx: RequestContext) -> None:
        # Pin the embedding identity FIRST. A project's vector space is decided
        # once, here, and every later embed (ingest and query) resolves through
        # it -- so it must exist before the namespace is shaped or anything is
        # written. For a project that already has a pin this just reads it back.
        identity = await ensure_project_pinned(ctx.account_id)

        # Apply the schema-bearing collection to the project's vector namespace
        # BEFORE creating preset directories. Directory init auto-generates L0/L1
        # sidecars whose embeddings are written to the vector store; if the
        # per-project adapter has not had its schema applied yet, those early
        # writes target a namespace with empty Fields and fail (logged as
        # "Collection ... does not exist"), losing the preset-dir/memory-scaffold
        # sidecars. Ensuring the collection first makes those writes succeed.
        #
        # The namespace is shaped from the PROJECT's dimension, not the live
        # config's: an older project keeps the vector width its data already
        # has, which is what lets a future model with a different dimension be
        # adopted by config alone.
        await service.vikingdb_manager.ensure_collection(ctx=ctx, identity=identity)
        await service.initialize_account_directories(ctx)
        await service.initialize_user_directories(ctx)

    return _init

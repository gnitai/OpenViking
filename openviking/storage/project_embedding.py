# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Durable per-project pinning of the embedding identity.

A project (== an OpenViking account) is stamped with an
:class:`EmbeddingIdentity` the first time it is touched, and keeps it for
life. Every embed -- ingestion and query alike -- resolves the embedder from
that pin instead of live config, which is what makes the global
``embedding.dense`` block mean "the default for NEW projects" rather than "the
model everything suddenly uses".

Why the pin lives here and not in the vector DB: the collection ``Description``
mechanism in ``collection_schemas`` cannot carry it. Turbopuffer (the
production backend) keeps ``Description`` in a per-process dict only --
``TurbopufferCollection.update()`` writes to memory and ``get_meta_data()``
merges just ``schema``/``distance_metric``/``approx_count`` back from the
remote -- so after a restart it always reads back as ``None``. AGFS is the only
durable per-account store, so pins go in ``_system`` alongside the existing
``users.json`` / ``setting.json``.

``_system`` is deliberate: it is filtered from listings by
``VikingFS._INTERNAL_NAMES`` and sits outside the indexed scope roots
(``user``/``agent``/``resources``/``session``), so a pin file never gets
indexed or triggers L0/L1 sidecar generation.
"""

import asyncio
import json
from typing import Any, Dict, Optional

from openviking.models.embedder.identity import (
    EmbeddingIdentity,
    default_identity,
    identity_for_unpinned,
)
from openviking_cli.utils import get_logger

logger = get_logger(__name__)

PIN_PATH_TEMPLATE = "/local/{account_id}/_system/embedding.json"


class ProjectEmbeddingPins:
    """Read/write per-account embedding pins, with a process-level cache."""

    def __init__(self, viking_fs: Any):
        from openviking.pyagfs.async_client import AsyncAGFSClient

        self._viking_fs = viking_fs
        self._async_agfs = AsyncAGFSClient(viking_fs.agfs)
        self._cache: Dict[str, EmbeddingIdentity] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ io

    async def _read_pin(self, account_id: str) -> Optional[EmbeddingIdentity]:
        path = PIN_PATH_TEMPLATE.format(account_id=account_id)
        try:
            content = await self._async_agfs.read(path)
            if isinstance(content, bytes):
                raw = content
            else:
                raw = content.content if hasattr(content, "content") else b""
            raw = await self._viking_fs.decrypt_bytes(account_id, raw)
            text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            payload = json.loads(text)
        except Exception:
            # Missing file and unreadable file are treated alike: the caller
            # falls back to the legacy identity, which is the safe direction
            # for a project that may already hold vectors.
            return None
        return EmbeddingIdentity.from_dict(payload)

    async def _write_pin(self, account_id: str, identity: EmbeddingIdentity) -> None:
        path = PIN_PATH_TEMPLATE.format(account_id=account_id)
        content = json.dumps(identity.to_dict(), ensure_ascii=False, indent=2).encode("utf-8")
        content = await self._viking_fs.encrypt_bytes(account_id, content)
        await self._ensure_parent_dirs(path)
        await self._async_agfs.write(path, content)

    async def _ensure_parent_dirs(self, path: str) -> None:
        parts = path.lstrip("/").split("/")
        for i in range(1, len(parts)):
            parent = "/" + "/".join(parts[:i])
            try:
                await self._async_agfs.mkdir(parent)
            except Exception:
                pass

    # ------------------------------------------------------------- resolve

    async def resolve(self, account_id: str) -> EmbeddingIdentity:
        """Return the identity this project's vectors live in.

        With no pin, the recorded eras decide -- never the live default on its
        own. An unpinned project predates pinning and is already full of an
        older model's vectors; resolving it to whatever the server currently
        defaults to would silently query a new model against old data.
        """
        cached = self._cache.get(account_id)
        if cached is not None:
            return cached

        identity = await self._read_pin(account_id)
        if identity is None:
            from openviking_cli.utils.config import get_openviking_config

            config = get_openviking_config()
            identity = identity_for_unpinned(config, account_id)
            logger.warning(
                "No embedding pin for account %s; resolving to %s from the recorded eras.",
                account_id,
                identity.model,
            )
        self._cache[account_id] = identity
        return identity

    async def ensure_pinned(self, account_id: str) -> EmbeddingIdentity:
        """Record the space this project is already in, if not recorded yet.

        Called from per-project init, which runs for EVERY account on first
        contact in a process -- not just new ones. So this cannot stamp the
        current default: an old project idle across the model change would
        have its first request after the flip write a new-model pin over a
        namespace full of old-model vectors, permanently, and the era cutoffs
        would never get a chance to fire. It writes what the eras say the
        account is, which for a genuinely new account IS the current default.

        Two concurrent first-requests may both write; the value is identical,
        so the race is benign.
        """
        cached = self._cache.get(account_id)
        if cached is not None:
            return cached

        async with self._lock:
            cached = self._cache.get(account_id)
            if cached is not None:
                return cached

            existing = await self._read_pin(account_id)
            if existing is not None:
                self._cache[account_id] = existing
                return existing

            from openviking_cli.utils.config import get_openviking_config

            identity = identity_for_unpinned(get_openviking_config(), account_id)
            try:
                await self._write_pin(account_id, identity)
            except Exception as err:
                # Do not cache on failure: a later request must retry, or the
                # project would run unpinned for the rest of the process and
                # resolve to legacy on the next restart.
                logger.error("Failed to write embedding pin for account %s: %s", account_id, err)
                raise
            logger.info(
                "Pinned account %s to embedding identity %s (dim=%d)",
                account_id,
                identity.model,
                identity.dimension,
            )
            self._cache[account_id] = identity
            return identity


_PINS: Optional[ProjectEmbeddingPins] = None


def init_project_embedding_pins(viking_fs: Any) -> ProjectEmbeddingPins:
    """Install the process-wide pin store."""
    global _PINS
    _PINS = ProjectEmbeddingPins(viking_fs)
    return _PINS


def get_project_embedding_pins() -> Optional[ProjectEmbeddingPins]:
    return _PINS


async def resolve_identity(account_id: str) -> EmbeddingIdentity:
    """Resolve an account's embedding identity, or the global default.

    The ``_PINS is None`` path covers embedded/library use where no VikingFS
    was built; there is no per-project storage to consult, so live config is
    all there is.
    """
    pins = _PINS
    if pins is None:
        from openviking_cli.utils.config import get_openviking_config

        return default_identity(get_openviking_config())
    return await pins.resolve(account_id)


async def ensure_project_pinned(account_id: str) -> EmbeddingIdentity:
    """Pin a project to the current default if it has no pin yet.

    Without a pin store (library use) there is nowhere to record the choice,
    so the live default stands in.
    """
    pins = _PINS
    if pins is None:
        from openviking_cli.utils.config import get_openviking_config

        return default_identity(get_openviking_config())
    return await pins.ensure_pinned(account_id)


async def resolve_embedder(account_id: str):
    """Return the embedder for ``account_id``'s pinned vector space."""
    from openviking_cli.utils.config import get_openviking_config

    identity = await resolve_identity(account_id)
    return get_openviking_config().embedding.get_embedder_for(identity)

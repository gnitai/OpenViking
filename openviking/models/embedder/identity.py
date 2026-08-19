# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Embedding identity: the vector-space a set of embeddings lives in.

A project (== an OpenViking account) is pinned to ONE embedding identity for
its whole life. Vectors produced by different models are not comparable, so
mixing them inside a namespace silently degrades retrieval with no error and
no log. Pinning makes the model a property of the project rather than of the
running server, which is what lets the global config change freely: the config
supplies the default for NEW projects only.

Credentials (``api_key``/``api_base``) are deliberately NOT part of the
identity -- they say where to reach a model, not which vector space it
produces. Keeping them out means key rotation and proxy moves stay global.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

# Frozen fallback used when a project predates per-project pinning AND the
# config carries no ``embedding.legacy`` block. Deliberately the model every
# such project was actually embedded with.
LEGACY_FALLBACK_PROVIDER = "openai"
LEGACY_FALLBACK_MODEL = "voyage/voyage-code-3"
LEGACY_FALLBACK_DIMENSION = 1024


@dataclass(frozen=True)
class EmbeddingIdentity:
    """The (provider, model, dimension) triple defining a vector space."""

    provider: str
    model: str
    dimension: int

    @property
    def cache_key(self) -> str:
        return f"{self.provider}|{self.model}|{self.dimension}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "dimension": self.dimension,
        }

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> Optional["EmbeddingIdentity"]:
        """Parse an identity, returning None when the payload is unusable.

        Returning None (rather than raising) lets callers treat a corrupt or
        truncated pin the same as a missing one and fall back to legacy, which
        is the safe direction for an already-indexed project.
        """
        if not isinstance(payload, dict):
            return None
        provider = payload.get("provider")
        model = payload.get("model")
        dimension = payload.get("dimension")
        if not provider or not model:
            return None
        try:
            dimension = int(dimension)
        except (TypeError, ValueError):
            return None
        if dimension <= 0:
            return None
        return cls(provider=str(provider).lower(), model=str(model), dimension=dimension)


def _active_dense_config(config: Any) -> Any:
    """Return the active embedding model config (hybrid > dense > sparse)."""
    embedding_cfg = config.embedding
    if embedding_cfg.hybrid is not None:
        return embedding_cfg.hybrid
    if embedding_cfg.dense is not None:
        return embedding_cfg.dense
    if embedding_cfg.sparse is not None:
        return embedding_cfg.sparse
    raise ValueError("No active embedding model configuration found")


def default_identity(config: Any) -> EmbeddingIdentity:
    """The identity NEW projects are pinned to, from live config.

    This is the ONLY place live config feeds an identity. Never use it as a
    fallback for an existing project -- see ``legacy_identity``.
    """
    model_cfg = _active_dense_config(config)
    provider = (
        getattr(model_cfg, "provider", None) or getattr(model_cfg, "backend", None) or ""
    ).lower()
    model = getattr(model_cfg, "model", None) or ""
    return EmbeddingIdentity(
        provider=provider,
        model=model,
        dimension=int(config.embedding.dimension),
    )


def _era_identity(era: Any) -> Optional[EmbeddingIdentity]:
    if era is None:
        return None
    return EmbeddingIdentity.from_dict(
        {
            "provider": era.provider,
            "model": era.model,
            "dimension": era.dimension,
        }
    )


def legacy_identity(config: Any) -> EmbeddingIdentity:
    """The oldest recorded era -- the space of the earliest projects.

    Read verbatim from the frozen ``embedding.legacy`` block. It is NEVER
    derived from live dense config: doing so would mean a future config-only
    model change silently re-pointed every unpinned legacy project at the new
    vector space -- exactly the corruption pinning exists to prevent.
    """
    eras = getattr(config.embedding, "legacy_eras", None)
    if eras:
        parsed = _era_identity(eras[0])
        if parsed is not None:
            return parsed
    legacy = getattr(config.embedding, "legacy", None)
    if legacy is not None and not isinstance(legacy, list):
        parsed = _era_identity(legacy)
        if parsed is not None:
            return parsed
    return EmbeddingIdentity(
        provider=LEGACY_FALLBACK_PROVIDER,
        model=LEGACY_FALLBACK_MODEL,
        dimension=LEGACY_FALLBACK_DIMENSION,
    )


def identity_for_unpinned(config: Any, account_id: str) -> EmbeddingIdentity:
    """The space an account with NO pin is already in.

    Walks the recorded eras oldest-first and takes the one the account falls
    into; an account past every era postdates them all, so the current default
    is what its vectors were made with.
    """
    era_for_unpinned = getattr(config.embedding, "era_for_unpinned", None)
    if era_for_unpinned is None:
        return legacy_identity(config)

    parsed = _era_identity(era_for_unpinned(account_id))
    if parsed is not None:
        return parsed
    # Past every recorded era: this account came into being under the current
    # default, so that is the space its vectors are in.
    return default_identity(config)

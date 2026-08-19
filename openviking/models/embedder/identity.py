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

    Never call this directly to decide an existing account's space -- go
    through ``identity_for_unpinned``, which consults the recorded eras first.
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


def identity_for_unpinned(config: Any, account_id: str) -> EmbeddingIdentity:
    """The space an account with NO pin is already in.

    Walks the recorded eras oldest-first and takes the one the account falls
    into; an account past every era postdates them all, so the current default
    is what its vectors were made with.

    Era identities are read verbatim and NEVER derived from live dense config:
    doing so would mean a later config-only model change silently re-pointed
    every unpinned legacy account at the new vector space -- exactly the
    corruption pinning exists to prevent.

    A deployment with no declared eras has never changed models, so there is no
    history to honour and the current default is right for everyone. This is
    why eras default to empty rather than to some particular model: guessing a
    previous model for an install that never had one would hand it an identity
    matching nothing it has ever written.
    """
    era_for_unpinned = getattr(config.embedding, "era_for_unpinned", None)
    if era_for_unpinned is not None:
        parsed = _era_identity(era_for_unpinned(account_id))
        if parsed is not None:
            return parsed
    # Past every recorded era: this account came into being under the current
    # default, so that is the space its vectors are in.
    return default_identity(config)

# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Per-project embedding pinning.

The guarantee under test: changing the global embedding model must move NEW
projects only. An already-indexed project keeps embedding and querying in the
vector space its stored vectors were built in, because mixing models inside a
namespace produces no error and no log -- just silently wrong neighbours.
"""

import json
from types import SimpleNamespace

import pytest

from openviking.models.embedder.identity import (
    EmbeddingIdentity,
    default_identity,
    legacy_identity,
)
from openviking.storage.project_embedding import PIN_PATH_TEMPLATE, ProjectEmbeddingPins


class _FakeSyncAGFS:
    """In-memory stand-in for the AGFS sync client."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.dirs: set[str] = set()

    def read(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    def write(self, path, data):
        self.files[path] = data
        return path

    def mkdir(self, path):
        self.dirs.add(path)
        return {}


class _FakeVikingFS:
    """VikingFS surface the pin store uses: the raw handle plus crypto."""

    def __init__(self, agfs):
        self.agfs = agfs

    async def encrypt_bytes(self, account_id, data):
        del account_id
        return data

    async def decrypt_bytes(self, account_id, data):
        del account_id
        return data


def _config(model="voyage/voyage-code-4", dimension=1024, legacy=None):
    """A config whose live default differs from the legacy block."""
    return SimpleNamespace(
        storage=SimpleNamespace(vectordb=SimpleNamespace(name="context", dimension=dimension)),
        embedding=SimpleNamespace(
            dimension=dimension,
            dense=SimpleNamespace(provider="openai", model=model, backend=None),
            sparse=None,
            hybrid=None,
            legacy=legacy
            or SimpleNamespace(
                provider="openai",
                model="voyage/voyage-code-3",
                dimension=1024,
            ),
        ),
    )


@pytest.fixture
def pins(monkeypatch):
    agfs = _FakeSyncAGFS()
    store = ProjectEmbeddingPins(_FakeVikingFS(agfs))
    monkeypatch.setattr(
        "openviking_cli.utils.config.get_openviking_config",
        lambda: _config(),
    )
    return store, agfs


# --------------------------------------------------------------- the trap


@pytest.mark.asyncio
async def test_unpinned_project_resolves_to_legacy_not_the_live_default(pins):
    """The regression that matters most.

    A project with no pin predates pinning, so it is already full of
    voyage-code-3 vectors. Resolving it to the LIVE default would embed its
    queries with code-4 against code-3 data -- no error, just degraded
    retrieval. It must land on legacy.
    """
    store, _ = pins

    identity = await store.resolve("legacy-project")

    assert identity.model == "voyage/voyage-code-3"
    assert identity.model != default_identity(_config()).model


@pytest.mark.asyncio
async def test_legacy_identity_ignores_live_dense_config():
    """The legacy block is frozen data, never derived from live config.

    If it inherited provider/dimension from `dense`, a LATER config-only model
    change would silently re-point every unpinned project at the new vector
    space -- the same corruption, one change later.
    """
    config = _config(model="something/else-v9", dimension=4096)

    identity = legacy_identity(config)

    assert identity.model == "voyage/voyage-code-3"
    assert identity.dimension == 1024
    assert identity.provider == "openai"


# ------------------------------------------------------------- new projects


@pytest.mark.asyncio
async def test_new_project_is_pinned_to_the_current_default(pins):
    store, agfs = pins

    identity = await store.ensure_pinned("brand-new")

    assert identity.model == "voyage/voyage-code-4"
    written = json.loads(agfs.files[PIN_PATH_TEMPLATE.format(account_id="brand-new")])
    assert written == {"provider": "openai", "model": "voyage/voyage-code-4", "dimension": 1024}


@pytest.mark.asyncio
async def test_pin_survives_a_later_default_change(pins, monkeypatch):
    """The whole point: flip the config, old projects do not move."""
    store, agfs = pins
    await store.ensure_pinned("early-adopter")

    # Config moves on to a different model, and a fresh process starts.
    monkeypatch.setattr(
        "openviking_cli.utils.config.get_openviking_config",
        lambda: _config(model="voyage/voyage-code-9"),
    )
    fresh = ProjectEmbeddingPins(_FakeVikingFS(agfs))

    assert (await fresh.resolve("early-adopter")).model == "voyage/voyage-code-4"
    assert (await fresh.ensure_pinned("newer-project")).model == "voyage/voyage-code-9"


@pytest.mark.asyncio
async def test_ensure_pinned_is_idempotent(pins):
    store, agfs = pins

    first = await store.ensure_pinned("acct")
    fresh = ProjectEmbeddingPins(_FakeVikingFS(agfs))
    second = await fresh.ensure_pinned("acct")

    assert first == second


@pytest.mark.asyncio
async def test_write_failure_is_not_cached(pins):
    """A failed write must not leave the project unpinned-but-cached.

    Caching a default on failure would run the project unpinned for the rest
    of the process and silently drop it to legacy on the next restart.
    """
    store, agfs = pins

    def _boom(path, data):
        raise OSError("disk full")

    agfs.write = _boom

    with pytest.raises(OSError):
        await store.ensure_pinned("doomed")
    assert "doomed" not in store._cache


# ------------------------------------------------------------- robustness


@pytest.mark.asyncio
async def test_corrupt_pin_falls_back_to_legacy(pins):
    """A truncated/corrupt pin is treated as missing, not as a crash.

    Falling back to legacy is the safe direction for a project that may
    already hold vectors.
    """
    store, agfs = pins
    agfs.files[PIN_PATH_TEMPLATE.format(account_id="corrupt")] = b"{not json"

    assert (await store.resolve("corrupt")).model == "voyage/voyage-code-3"


def test_identity_rejects_incomplete_payloads():
    assert EmbeddingIdentity.from_dict(None) is None
    assert EmbeddingIdentity.from_dict({"model": "m"}) is None
    assert EmbeddingIdentity.from_dict({"provider": "p", "model": "m"}) is None
    assert EmbeddingIdentity.from_dict({"provider": "p", "model": "m", "dimension": 0}) is None
    assert EmbeddingIdentity.from_dict(
        {"provider": "OpenAI", "model": "m", "dimension": "1024"}
    ) == EmbeddingIdentity(provider="openai", model="m", dimension=1024)


# ------------------------------------------------- mixed fleet, one process


@pytest.mark.asyncio
async def test_one_handler_embeds_each_project_with_its_own_model(monkeypatch):
    """The core guarantee, at the ingestion path.

    A single TextEmbeddingHandler drains the queue for every account. With the
    default flipped to code-4, a legacy project's NEW chunks must still be
    embedded with code-3, or that project's namespace ends up holding two
    incompatible vector spaces.
    """
    from openviking.storage import project_embedding
    from openviking.storage.collection_schemas import TextEmbeddingHandler

    agfs = _FakeSyncAGFS()
    store = project_embedding.init_project_embedding_pins(_FakeVikingFS(agfs))
    try:
        built = []

        def _get_embedder_for(identity):
            built.append(identity.model)
            return SimpleNamespace(model=identity.model, is_sparse=False)

        config = _config()
        config.embedding.get_embedder = lambda: SimpleNamespace(is_sparse=False)
        config.embedding.get_embedder_for = _get_embedder_for
        config.embedding.circuit_breaker = SimpleNamespace(
            failure_threshold=5, reset_timeout=60.0, max_reset_timeout=600.0
        )
        monkeypatch.setattr(
            "openviking_cli.utils.config.get_openviking_config",
            lambda: config,
        )

        # "old" predates pinning; "new" is stamped with the current default.
        await store.ensure_pinned("new-project")

        handler = TextEmbeddingHandler(SimpleNamespace(is_closing=False))

        old_embedder, old_dim = await handler._resolve_for_account("old-project")
        new_embedder, new_dim = await handler._resolve_for_account("new-project")

        assert old_embedder.model == "voyage/voyage-code-3"
        assert new_embedder.model == "voyage/voyage-code-4"
        assert old_dim == new_dim == 1024
        assert built == ["voyage/voyage-code-3", "voyage/voyage-code-4"]
    finally:
        project_embedding._PINS = None


@pytest.mark.asyncio
async def test_handler_without_pin_store_uses_its_own_embedder(monkeypatch):
    """Library/embedded use has no per-project storage to consult."""
    from openviking.storage import project_embedding
    from openviking.storage.collection_schemas import TextEmbeddingHandler

    project_embedding._PINS = None
    own = SimpleNamespace(model="whatever", is_sparse=False)
    config = _config()
    config.embedding.get_embedder = lambda: own
    config.embedding.circuit_breaker = SimpleNamespace(
        failure_threshold=5, reset_timeout=60.0, max_reset_timeout=600.0
    )
    monkeypatch.setattr(
        "openviking_cli.utils.config.get_openviking_config",
        lambda: config,
    )

    handler = TextEmbeddingHandler(SimpleNamespace(is_closing=False))
    embedder, dim = await handler._resolve_for_account("anything")

    assert embedder is own
    assert dim == 1024

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
    identity_for_unpinned,
)
from openviking.storage.project_embedding import PIN_PATH_TEMPLATE, ProjectEmbeddingPins

# Accounts below this id predate the model change; at or above it they are new.
LEGACY_CUTOFF = 168794
OLD_ACCOUNT = "100"
NEW_ACCOUNT = "999999"


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


def _config(model="voyage/voyage-code-4", dimension=1024, eras=None):
    """A mutable config stub whose live default differs from its one era.

    The era objects and the selection logic are the REAL ones -- only the
    surrounding container is a namespace, so individual tests can swap in fake
    embedder factories.
    """
    from openviking_cli.utils.config.embedding_config import LegacyEmbeddingIdentityConfig

    if eras is None:
        # Mirrors the deployed shape: one prior era, bounded by an id cutoff.
        # Accounts below LEGACY_CUTOFF are old; at or above it, new.
        eras = [
            LegacyEmbeddingIdentityConfig(
                provider="openai",
                model="voyage/voyage-code-3",
                dimension=1024,
                applies_below_account_id=LEGACY_CUTOFF,
            )
        ]
    ordered = sorted(eras, key=lambda era: era.sort_key)

    def _era_for_unpinned(account_id):
        for era in ordered:
            if era.covers_account(account_id):
                return era
        return None

    return SimpleNamespace(
        storage=SimpleNamespace(vectordb=SimpleNamespace(name="context", dimension=dimension)),
        embedding=SimpleNamespace(
            dimension=dimension,
            dense=SimpleNamespace(provider="openai", model=model, backend=None),
            sparse=None,
            hybrid=None,
            legacy=ordered,
            legacy_eras=ordered,
            era_for_unpinned=_era_for_unpinned,
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


def test_era_identities_ignore_live_dense_config():
    """Eras are frozen data, never derived from live config.

    If an era inherited provider/dimension from `dense`, a LATER config-only
    model change would silently re-point every unpinned project at the new
    vector space -- the same corruption, one change later.
    """
    config = _config(model="something/else-v9", dimension=4096)

    identity = identity_for_unpinned(config, "any-account")

    assert identity.model == "voyage/voyage-code-3"
    assert identity.dimension == 1024
    assert identity.provider == "openai"


def test_no_declared_eras_means_the_current_default():
    """A deployment that never changed models has no history to honour.

    Defaulting eras to some particular model would hand a fresh install an
    identity matching nothing it has ever written -- and, worse, one built
    with another provider's credentials.
    """
    config = _real_config("bge-small-zh-v1.5-f16", [])

    identity = identity_for_unpinned(config, "42")

    assert identity.model == "bge-small-zh-v1.5-f16"


# ------------------------------------------------------------- new projects


@pytest.mark.asyncio
async def test_new_project_is_pinned_to_the_current_default(pins):
    store, agfs = pins

    identity = await store.ensure_pinned(NEW_ACCOUNT)

    assert identity.model == "voyage/voyage-code-4"
    written = json.loads(agfs.files[PIN_PATH_TEMPLATE.format(account_id=NEW_ACCOUNT)])
    assert written == {"provider": "openai", "model": "voyage/voyage-code-4", "dimension": 1024}


@pytest.mark.asyncio
async def test_pin_survives_a_later_default_change(pins, monkeypatch):
    """The whole point: flip the config, old projects do not move."""
    store, agfs = pins
    await store.ensure_pinned(NEW_ACCOUNT)

    # Config moves on to a different model, and a fresh process starts.
    monkeypatch.setattr(
        "openviking_cli.utils.config.get_openviking_config",
        lambda: _config(model="voyage/voyage-code-9"),
    )
    fresh = ProjectEmbeddingPins(_FakeVikingFS(agfs))

    assert (await fresh.resolve(NEW_ACCOUNT)).model == "voyage/voyage-code-4"
    assert (await fresh.ensure_pinned("999998")).model == "voyage/voyage-code-9"


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
        await store.ensure_pinned(NEW_ACCOUNT)

        handler = TextEmbeddingHandler(SimpleNamespace(is_closing=False))

        old_embedder, old_dim = await handler._resolve_for_account(OLD_ACCOUNT)
        new_embedder, new_dim = await handler._resolve_for_account(NEW_ACCOUNT)

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


# ------------------------------------------------------------- query path


@pytest.mark.asyncio
async def test_query_is_embedded_with_the_projects_pinned_model(monkeypatch):
    """The path that degrades silently if it is missed.

    A code-4 query vector searched against code-3 stored vectors returns
    neighbours with no error and no log. An un-migrated project must keep
    querying with the model its vectors were built with.
    """
    from openviking.retrieve.hierarchical_retriever import HierarchicalRetriever
    from openviking.storage import project_embedding

    agfs = _FakeSyncAGFS()
    store = project_embedding.init_project_embedding_pins(_FakeVikingFS(agfs))
    try:
        config = _config()
        config.embedding.get_embedder_for = lambda identity: SimpleNamespace(model=identity.model)
        monkeypatch.setattr(
            "openviking_cli.utils.config.get_openviking_config",
            lambda: config,
        )
        await store.ensure_pinned(NEW_ACCOUNT)

        default_embedder = SimpleNamespace(model="server-default")
        retriever = HierarchicalRetriever(storage=None, embedder=default_embedder)

        old = await retriever._resolve_query_embedder(SimpleNamespace(account_id=OLD_ACCOUNT))
        new = await retriever._resolve_query_embedder(SimpleNamespace(account_id=NEW_ACCOUNT))

        assert old.model == "voyage/voyage-code-3"
        assert new.model == "voyage/voyage-code-4"
    finally:
        project_embedding._PINS = None


@pytest.mark.asyncio
async def test_query_keeps_caller_embedder_without_a_pin_store():
    """Library use has no per-project storage; the caller's embedder stands."""
    from openviking.retrieve.hierarchical_retriever import HierarchicalRetriever
    from openviking.storage import project_embedding

    project_embedding._PINS = None
    own = SimpleNamespace(model="caller-supplied")
    retriever = HierarchicalRetriever(storage=None, embedder=own)

    resolved = await retriever._resolve_query_embedder(SimpleNamespace(account_id="whatever"))

    assert resolved is own


# ------------------------------------------------------------------ ovpack


def test_ovpack_labels_exports_with_the_projects_model(monkeypatch):
    """An export must record the space its vectors are actually in.

    Labelling from live config would stamp a code-3 project's export as
    code-4, making it non-restorable into the very project it came from.
    """
    from openviking.storage.ovpack.vectors import (
        current_embedding_metadata,
        embedding_snapshot_metadata,
    )

    config = _config()
    config.embedding.dense.input = "text"
    config.embedding.dense.query_param = None
    config.embedding.dense.document_param = None
    config.embedding.dense.get_effective_dimension = lambda: 1024
    monkeypatch.setattr(
        "openviking_cli.utils.config.get_openviking_config",
        lambda: config,
    )
    legacy = EmbeddingIdentity(provider="openai", model="voyage/voyage-code-3", dimension=1024)

    # Live config alone would say code-4 for both.
    assert embedding_snapshot_metadata(None)["model"] == "voyage/voyage-code-4"
    assert embedding_snapshot_metadata(None, legacy)["model"] == "voyage/voyage-code-3"
    assert current_embedding_metadata(legacy)["model"] == "voyage/voyage-code-3"
    assert current_embedding_metadata(legacy)["dimensions"] == 1024


# ------------------------------------------------------- account-id cutoff


def _real_config(default_model, eras):
    """A REAL EmbeddingConfig, so era resolution runs the shipped logic.

    ``eras`` is a list of (model, applies_below_account_id).
    """
    from openviking_cli.utils.config.embedding_config import EmbeddingConfig

    embedding = EmbeddingConfig(
        dense={
            "provider": "openai",
            "model": default_model,
            "dimension": 1024,
            "api_key": "test",
        },
        legacy=[
            {
                "provider": "openai",
                "model": model,
                "dimension": 1024,
                "applies_below_account_id": cutoff,
            }
            for model, cutoff in eras
        ],
    )
    return SimpleNamespace(embedding=embedding)


def _cutoff_config(cutoff):
    """One legacy era, optionally bounded by a numeric account-id cutoff."""
    return _real_config("voyage/voyage-code-4", [("voyage/voyage-code-3", cutoff)])


@pytest.mark.asyncio
async def test_account_id_cutoff_splits_old_from_new(monkeypatch):
    """A numeric cutoff replaces the backfill for a fleet with ordered ids."""
    agfs = _FakeSyncAGFS()
    store = ProjectEmbeddingPins(_FakeVikingFS(agfs))
    monkeypatch.setattr(
        "openviking_cli.utils.config.get_openviking_config",
        lambda: _cutoff_config(168794),
    )

    assert (await store.resolve("168793")).model == "voyage/voyage-code-3"
    assert (await store.resolve("168794")).model == "voyage/voyage-code-4"
    assert (await store.resolve("200000")).model == "voyage/voyage-code-4"


@pytest.mark.asyncio
async def test_non_numeric_accounts_fall_to_legacy_under_a_cutoff(monkeypatch):
    """Legacy is the safe direction for an id the cutoff cannot rank."""
    agfs = _FakeSyncAGFS()
    store = ProjectEmbeddingPins(_FakeVikingFS(agfs))
    monkeypatch.setattr(
        "openviking_cli.utils.config.get_openviking_config",
        lambda: _cutoff_config(168794),
    )

    assert (await store.resolve("acme-corp")).model == "voyage/voyage-code-3"


@pytest.mark.asyncio
async def test_pin_beats_the_cutoff(monkeypatch):
    """A written pin is the truth; the cutoff is only a fallback for its absence."""
    agfs = _FakeSyncAGFS()
    store = ProjectEmbeddingPins(_FakeVikingFS(agfs))
    monkeypatch.setattr(
        "openviking_cli.utils.config.get_openviking_config",
        lambda: _cutoff_config(168794),
    )
    # An id far above the cutoff, but pinned to the old space.
    await store._write_pin(
        "999999",
        EmbeddingIdentity(provider="openai", model="voyage/voyage-code-3", dimension=1024),
    )

    assert (await store.resolve("999999")).model == "voyage/voyage-code-3"


@pytest.mark.asyncio
async def test_no_cutoff_treats_every_unpinned_account_as_legacy(monkeypatch):
    agfs = _FakeSyncAGFS()
    store = ProjectEmbeddingPins(_FakeVikingFS(agfs))
    monkeypatch.setattr(
        "openviking_cli.utils.config.get_openviking_config",
        lambda: _cutoff_config(None),
    )

    assert (await store.resolve("999999")).model == "voyage/voyage-code-3"


@pytest.mark.asyncio
async def test_three_models_need_only_one_appended_era(monkeypatch):
    """Changing the model a second time appends an era; it never rewrites one.

    Eras describe frozen history, so the code-3 entry written at the first
    change stays byte-identical forever. Only the new boundary is new.
    """
    agfs = _FakeSyncAGFS()
    store = ProjectEmbeddingPins(_FakeVikingFS(agfs))
    monkeypatch.setattr(
        "openviking_cli.utils.config.get_openviking_config",
        lambda: _real_config(
            "voyage/voyage-code-5",
            [("voyage/voyage-code-3", 168794), ("voyage/voyage-code-4", 250000)],
        ),
    )

    assert (await store.resolve("100")).model == "voyage/voyage-code-3"
    assert (await store.resolve("168793")).model == "voyage/voyage-code-3"
    assert (await store.resolve("168794")).model == "voyage/voyage-code-4"
    assert (await store.resolve("249999")).model == "voyage/voyage-code-4"
    # Past every recorded era -> it was born under the current default.
    assert (await store.resolve("250000")).model == "voyage/voyage-code-5"
    # Unrankable ids fall to the OLDEST era, the safe direction.
    assert (await store.resolve("acme-corp")).model == "voyage/voyage-code-3"


@pytest.mark.asyncio
async def test_eras_are_sorted_not_trusted_in_file_order(monkeypatch):
    """An era appended out of order still lands in the right place."""
    agfs = _FakeSyncAGFS()
    store = ProjectEmbeddingPins(_FakeVikingFS(agfs))
    monkeypatch.setattr(
        "openviking_cli.utils.config.get_openviking_config",
        lambda: _real_config(
            "voyage/voyage-code-5",
            [("voyage/voyage-code-4", 250000), ("voyage/voyage-code-3", 168794)],
        ),
    )

    assert (await store.resolve("100")).model == "voyage/voyage-code-3"
    assert (await store.resolve("200000")).model == "voyage/voyage-code-4"


@pytest.mark.asyncio
async def test_ensure_pinned_does_not_stamp_a_legacy_project_with_the_new_model(monkeypatch):
    """The bug this whole design exists to prevent, on the path that runs.

    Per-project init calls ensure_pinned for EVERY account on first contact in
    a process, not just new ones. A project idle across the model change is
    still unpinned when its next request lands; if ensure_pinned wrote the
    current default, that request would permanently stamp code-4 onto a
    namespace full of code-3 vectors and the era cutoff would never fire.
    """
    agfs = _FakeSyncAGFS()
    store = ProjectEmbeddingPins(_FakeVikingFS(agfs))
    monkeypatch.setattr(
        "openviking_cli.utils.config.get_openviking_config",
        lambda: _real_config("voyage/voyage-code-4", [("voyage/voyage-code-3", 168794)]),
    )

    old = await store.ensure_pinned("100")
    new = await store.ensure_pinned("999999")

    assert old.model == "voyage/voyage-code-3", "a legacy account must not be re-pointed"
    assert new.model == "voyage/voyage-code-4"
    # And it is durable, not just an in-memory answer.
    written = json.loads(agfs.files[PIN_PATH_TEMPLATE.format(account_id="100")])
    assert written["model"] == "voyage/voyage-code-3"


@pytest.mark.asyncio
async def test_ensure_pinned_and_resolve_agree(monkeypatch):
    """Whichever path touches an account first, it lands in the same space.

    A crash-recovered queue message reaches resolve(); an HTTP request reaches
    ensure_pinned(). If they disagreed, an account's identity would depend on
    which arrived first.
    """
    config = _real_config("voyage/voyage-code-4", [("voyage/voyage-code-3", 168794)])
    monkeypatch.setattr("openviking_cli.utils.config.get_openviking_config", lambda: config)

    for account_id in ("100", "168793", "168794", "999999", "acme-corp"):
        via_resolve = await ProjectEmbeddingPins(_FakeVikingFS(_FakeSyncAGFS())).resolve(account_id)
        via_pin = await ProjectEmbeddingPins(_FakeVikingFS(_FakeSyncAGFS())).ensure_pinned(
            account_id
        )
        assert via_resolve == via_pin, f"paths disagree for {account_id}"

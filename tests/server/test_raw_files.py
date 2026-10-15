"""Raw file store: PUT /api/v1/raw-files writes into the un-indexed `files` scope."""

import asyncio

import httpx
import pytest

from openviking.server.identity import RequestContext, Role
from openviking.storage.queuefs.embedding_queue import EmbeddingQueue
from openviking.storage.queuefs.semantic_queue import SemanticQueue
from openviking.storage.viking_fs import VikingFS
from openviking_cli.exceptions import InvalidArgumentError
from openviking_cli.session.user_id import UserIdentifier

PDF = b"%PDF-1.4\n\xe2\xe3\xcf\xd3 binary \xff\xfe\x00\x01 %%EOF"
URI = "wfs://files/sessions/s1/uploads/report.pdf"


async def test_put_stores_raw_bytes(client):
    r = await client.put("/api/v1/raw-files", params={"uri": URI}, content=PDF)

    assert r.status_code == 200
    assert r.json()["result"] == {"uri": URI, "size": len(PDF)}
    got = await client.get("/api/v1/content/download", params={"uri": URI})
    assert got.status_code == 200
    assert got.content == PDF


@pytest.mark.parametrize(
    "uri",
    ["wfs://resources/notes.txt", "wfs://files", "wfs://files/", "s3://files/x.txt"],
)
async def test_put_refuses_uri_outside_files(client, service, uri):
    r = await client.put("/api/v1/raw-files", params={"uri": uri}, content=b"x")

    assert r.status_code == 400
    assert not await service.viking_fs.exists("wfs://resources/notes.txt")


async def test_put_refuses_taken_name(client):
    await client.put("/api/v1/raw-files", params={"uri": URI}, content=PDF)

    r = await client.put("/api/v1/raw-files", params={"uri": URI}, content=b"other")

    assert r.status_code == 409
    got = await client.get("/api/v1/content/download", params={"uri": URI})
    assert got.content == PDF


async def test_concurrent_puts_of_one_name_store_once(client, monkeypatch):
    real_write = VikingFS.write_file_bytes

    async def slow_write(self, *args, **kwargs):
        await asyncio.sleep(0.2)
        return await real_write(self, *args, **kwargs)

    monkeypatch.setattr(VikingFS, "write_file_bytes", slow_write)

    a, b = await asyncio.gather(
        client.put("/api/v1/raw-files", params={"uri": URI}, content=b"a"),
        client.put("/api/v1/raw-files", params={"uri": URI}, content=b"b"),
    )

    assert sorted([a.status_code, b.status_code]) == [200, 409]
    winner = b"a" if a.status_code == 200 else b"b"
    got = await client.get("/api/v1/content/download", params={"uri": URI})
    assert got.content == winner


async def test_dir_mode_claims_a_folder_once(client, service):
    folder = "wfs://files/sessions/s1/uploads/project"
    params = {"uri": folder + "/", "dir": "true"}

    first = await client.put("/api/v1/raw-files", params=params)
    second = await client.put("/api/v1/raw-files", params=params)

    assert first.status_code == 200
    assert first.json()["result"] == {"uri": folder}
    assert (await service.viking_fs.stat(folder))["isDir"]
    assert second.status_code == 409


async def test_dir_mode_refuses_a_body(client, service):
    folder = "wfs://files/sessions/s1/uploads/project"

    r = await client.put("/api/v1/raw-files", params={"uri": folder, "dir": "true"}, content=b"x")

    assert r.status_code == 400
    assert not await service.viking_fs.exists(folder)


async def _chunks(total: int, size: int = 1024 * 1024):
    while total > 0:
        yield b"x" * min(size, total)
        total -= size


@pytest.mark.parametrize("chunked", [False, True], ids=["content-length", "chunked"])
async def test_put_refuses_body_over_20_mb(client, service, chunked):
    over = 20 * 1024 * 1024 + 1
    content = _chunks(over) if chunked else b"x" * over

    r = await client.put("/api/v1/raw-files", params={"uri": URI}, content=content)

    assert r.status_code == 413
    assert not await service.viking_fs.exists(URI)


async def test_put_takes_exactly_20_mb(client):
    r = await client.put(
        "/api/v1/raw-files", params={"uri": URI}, content=_chunks(20 * 1024 * 1024)
    )

    assert r.status_code == 200
    assert r.json()["result"]["size"] == 20 * 1024 * 1024


async def test_concurrent_puts_of_different_names_into_a_new_folder_all_store(client):
    names = [f"wfs://files/sessions/s2/uploads/f{i}.txt" for i in range(10)]

    rs = await asyncio.gather(
        *(client.put("/api/v1/raw-files", params={"uri": u}, content=b"x") for u in names)
    )

    assert [r.status_code for r in rs] == [200] * len(names)


async def test_files_scope_stays_unindexed_unlisted_and_out_of_packs(
    client, service, sample_markdown_file, monkeypatch, tmp_path
):
    enqueued = []
    real_sem, real_emb = SemanticQueue.enqueue, EmbeddingQueue.enqueue

    async def spy_sem(self, msg):
        enqueued.append(msg.uri)
        return await real_sem(self, msg)

    async def spy_emb(self, msg):
        if msg is not None:
            enqueued.append((msg.context_data or {}).get("uri", ""))
        return await real_emb(self, msg)

    monkeypatch.setattr(SemanticQueue, "enqueue", spy_sem)
    monkeypatch.setattr(EmbeddingQueue, "enqueue", spy_emb)
    ctx = RequestContext(user=UserIdentifier.the_default_user(), role=Role.ROOT)

    notes = "wfs://files/sessions/s1/uploads/notes.md"
    await client.put("/api/v1/raw-files", params={"uri": notes}, content=b"# Notes\n\nneedle\n")
    await client.put(
        "/api/v1/raw-files", params={"uri": "wfs://files/sessions/s1/uploads/zip", "dir": "true"}
    )
    # A Knowledge add elsewhere (refreshes parents up to wfs://resources), then reindex resources.
    await service.resources.add_resource(
        path=str(sample_markdown_file),
        ctx=ctx,
        to="wfs://resources/kb/sample",
        reason="t",
        wait=True,
    )
    for mode in ["vectors_only", "semantic_and_vectors"]:
        r = await client.post(
            "/api/v1/content/reindex", json={"uri": "wfs://resources", "mode": mode, "wait": True}
        )
        assert r.status_code == 200
    await service.resources.wait_processed(timeout=30)

    rows, _ = await service.vikingdb_manager.scroll(limit=10000, output_fields=["uri"], ctx=ctx)
    vector_uris = {row.get("uri", "") for row in rows}
    assert any(u.startswith("wfs://resources/kb") for u in vector_uris)  # control: indexing ran
    assert not [u for u in vector_uris if u.startswith("wfs://files")]
    assert not [u for u in enqueued if u and u.startswith("wfs://files")]
    for folder in ["wfs://files/sessions/s1", "wfs://files/sessions/s1/uploads"]:
        assert not await service.viking_fs.exists(f"{folder}/.abstract.md")
        assert not await service.viking_fs.exists(f"{folder}/.overview.md")

    root = await client.get("/api/v1/fs/ls", params={"uri": "wfs://", "simple": True})
    assert root.status_code == 200
    assert not [u for u in root.json()["result"] if "files" in u]

    with pytest.raises(InvalidArgumentError, match="not supported for scope: files"):
        await service.pack.export_ovpack(
            "wfs://files/sessions/s1", str(tmp_path / "s1.ovpack"), ctx=ctx
        )


@pytest.mark.parametrize(
    "uri",
    [
        "wfs://files/../resources/evil.txt",
        "wfs://files/sessions/../../resources/evil.txt",
        "wfs://files/sessions/./s1/evil.txt",
    ],
)
async def test_put_refuses_dot_segments(client, service, uri):
    r = await client.put("/api/v1/raw-files", params={"uri": uri}, content=b"x")

    assert r.status_code == 403  # VikingFS's own traversal guard
    assert not await service.viking_fs.exists("wfs://resources/evil.txt")
    assert not await service.viking_fs.exists("wfs://files/sessions/s1/evil.txt")


async def test_put_refuses_when_the_name_cannot_be_checked(app, client, monkeypatch):
    await client.put("/api/v1/raw-files", params={"uri": URI}, content=PDF)
    real_stat = VikingFS.stat

    async def broken_stat(self, *args, **kwargs):
        raise RuntimeError("storage unavailable")

    monkeypatch.setattr(VikingFS, "stat", broken_stat)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        r = await c.put("/api/v1/raw-files", params={"uri": URI}, content=b"other")
    monkeypatch.setattr(VikingFS, "stat", real_stat)

    assert r.status_code >= 500
    got = await client.get("/api/v1/content/download", params={"uri": URI})
    assert got.content == PDF

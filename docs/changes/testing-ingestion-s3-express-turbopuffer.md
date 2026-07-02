# Testing ingestion against S3 Express + Turbopuffer

A local, end-to-end smoke test that exercises the fork's storage keepers:
S3 Express One Zone storage, the Turbopuffer vector backend, async locks, and
the S3 Express **native rename** path. Use it to validate a build (e.g. after a
merge) before promoting it.

Related: [`s3-express-and-lock-speedups.md`](./s3-express-and-lock-speedups.md).

## Prerequisites

- A config file with `agfs.backend=s3` + `s3.express=true` and
  `vectordb.backend=turbopuffer`. Example in the repo root:
  [`ov-express-turbopuffer.conf`](../../ov-express-turbopuffer.conf).
- **Turbopuffer API key.** The `turbopuffer` config block usually has *no* key
  inline — the adapter reads `TURBOPUFFER_API_KEY` from the environment
  (`turbopuffer_adapter.py`). If it is unset the server boots fine but every
  vector op fails at ingest time. Export it before starting the server.
- Python deps synced: `uv sync --extra test --extra dev`.

## 1. Pre-flight: build the right binaries

The Python server loads the Rust S3 backend from the prebuilt native extension
`openviking/lib/ragfs_python.abi3.so` (sources: `crates/ragfs`,
`crates/ragfs-python`). The `ov` CLI runs `./target/release/ov` (sources:
`crates/ov_cli`). A stale binary means you test the wrong code.

- If `crates/ragfs*` changed, rebuild the extension (your maturin / `uv sync` path).
- If `crates/ov_cli` changed, rebuild the client:

```bash
cargo build --release -p ov_cli
```

Quick check: compare the binary mtimes against the latest commit touching their
sources; rebuild if older.

## 2. Start the server

The example config has no `server` block, so it defaults to `127.0.0.1:1933`
and (with no `root_api_key`) `auth_mode=dev`.

```bash
TURBOPUFFER_API_KEY=<your-tpuf-key> \
  uv run openviking-server --config ov-express-turbopuffer.conf --port 1933 \
  > /tmp/ov-server.log 2>&1 &
```

Wait for health, then confirm the backend mode:

```bash
until curl -sf http://127.0.0.1:1933/health >/dev/null; do sleep 2; done
curl -s http://127.0.0.1:1933/health   # {"status":"ok",...,"auth_mode":"dev"}
grep "mode=turbopuffer" /tmp/ov-server.log   # confirms vector backend, not local fallback
```

> **Readiness vs health.** `/health` goes green at *application startup*, before the
> *deferred* service init finishes (embedder, collection backfill, resource
> processor). Ingesting too early returns
> `[NOT_INITIALIZED] Resourceprocessor not initialized`. Wait for the real
> ready marker before issuing requests:
>
> ```bash
> until grep -q "OpenVikingService initialized" /tmp/ov-server.log; do sleep 1; done
> ```
>
> A first run against an existing collection also backfills embedding metadata
> ("Existing collection has N vector(s) but no embedding metadata ... Backfilling")
> — this can add ~15-20s before ready.

## 3. Point the CLI at the local server

The default `~/.openviking/ovcli.conf` may point at a remote deployment. Use a
local one via `OPENVIKING_CLI_CONFIG_FILE` (in DEV mode the `api_key` is not
validated):

```jsonc
// ~/.openviking/ovcli-local.conf
{
  "url": "http://127.0.0.1:1933",
  "api_key": "dev",
  "account": "default",
  "user": "default",
  "timeout": 120.0,
  "output": "table"
}
```

```bash
export OPENVIKING_CLI_CONFIG_FILE=~/.openviking/ovcli-local.conf
uv run ov ls /   # sanity: lists top-level scopes
```

## 4. Ingest a small, fresh corpus

Use a **unique target URI** so the result is unambiguous and you don't hit the
update path by accident. Keep it small — embedding + VLM are live endpoints.

```bash
mkdir -p /tmp/ov-ingest-test
printf 'def add(a, b):\n    return a + b\n' > /tmp/ov-ingest-test/calculator.py
echo '# Ingest smoke test corpus' > /tmp/ov-ingest-test/README.md

uv run ov add-resource /tmp/ov-ingest-test \
  --parent-auto-create wfs://resources/smoke-tpuf-$(date +%H%M) \
  --wait --timeout 300 --reason "smoke test"
```

A clean run ends with `status: success` and a `queue_status` showing
`Semantic`/`Embedding` `processed` counts with `error_count: 0`.

> **`--wait` client error on directory ingest.** Ingesting a *directory* with
> `--wait` reproducibly returns a client-side
> `Network error: ... error sending request for url (.../api/v1/resources)`,
> while the **server still completes the ingest in full**. Single-file `--wait`
> does not show this. Treat the client error as cosmetic and confirm success in
> the server log instead:
>
> ```bash
> grep -E "Completed semantic generation|All embedding tasks\([0-9]+\) completed" /tmp/ov-server.log
> ```
>
> If both appear for your ingest's `SemanticMsg` id, the data landed regardless
> of the client error. (The `find` verification in step 5 is the real proof.)

## 5. Verify the keepers (don't trust exit code alone)

**Vectors reached Turbopuffer** — the ingested files come back from semantic search:

```bash
uv run ov find "divide two numbers" | grep smoke-tpuf
```

If `find` is empty, embeddings did not land (re-check `TURBOPUFFER_API_KEY`).

**S3 Express native rename** — move a file, then inspect the `[mv]` log line:

```bash
uv run ov mv wfs://resources/smoke-tpuf-XXXX/ov-ingest-test/calculator.py \
            wfs://resources/smoke-tpuf-XXXX/ov-ingest-test/calc_renamed.py
grep '\[mv\]' /tmp/ov-server.log
```

Look for `rm_ms=0` — the source-delete is skipped only when
`_used_native_mv=True` (`viking_fs.py`), i.e. the native `RenameObject` path was
taken instead of copy+delete. `cp_ms` is then the native rename itself, and
`vec_ms>0` shows the vector index was updated for the moved file.

**Vector upsert-on-move** — after the rename, the new path is searchable and the
old one is gone (no stale duplicate):

```bash
uv run ov find "calculator add divide" | grep -E 'calc_renamed|calculator\.py'
# expect calc_renamed.py present, calculator.py absent
```

**Async locks / no errors** — during ingest the log shows `lock_acquire_ms`
instrumentation and sidecar writes, with no tracebacks.

## 6. Clean up

```bash
uv run ov rm wfs://resources/smoke-tpuf-XXXX   # remove test data from the real bucket + Turbopuffer
pkill -f "openviking-server --config ov-express-turbopuffer"
```

Test data lives in the **real** S3 Express bucket and the Turbopuffer `context`
collection — remove it so it doesn't accumulate alongside production vectors.

### Stale data-directory lock

The server holds a single-instance lock at
`<workspace>/.openviking.pid`. A clean exit removes it; a `kill -9` (or a crash)
leaves it behind, and the next start fails with
`DataDirectoryLocked: Another OpenViking process (PID …) is already using the
data directory`. If no such process is actually running, remove the stale file:

```bash
PID=$(cat <workspace>/.openviking.pid)
kill -0 "$PID" 2>/dev/null || rm -f <workspace>/.openviking.pid   # only if PID is dead
```

### Shutdown-log noise (expected)

Stopping the server with `pkill`/SIGTERM produces a traceback ending in
`SystemExit: 0` plus warnings like `Event loop is closed` and
`cannot schedule new futures after shutdown`. These are teardown races from the
SIGTERM handler unwinding the event loop mid-flight — harmless, and they do not
affect persisted data. A clean `SIGINT` (Ctrl-C in the foreground) avoids most
of the noise.

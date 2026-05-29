# Upstream sync summary (`volcengine/OpenViking` main → fork)

What this sync pulled in, and what was changed by hand to make it land on top of
the fork's customizations.

- **Merge base:** `dbb67f75` (where the fork and upstream last agreed)
- **Upstream tip:** `cb3e78ce`
- **Commits pulled:** 76 non-merge commits (≈ "78" including merges)
- **Total upstream churn:** 421 files, +25k / −23k lines
- **Result:** merge commit `d5535881` on branch `sync-upstream-main`
- **Follow-up:** `90e77bca` (Turbopuffer review fixes), described at the end

Commit mix: **30 fixes, 19 docs, 13 features, 4 refactors, 1 chore, 1 benchmark.**

---

## Part 1 — What upstream changed (the 76 commits)

### 1. The big one: storage made fully async (#2143)

`refactor(storage): 异步化存储锁与 IO` ("asyncify storage locks and IO") — 46
files, the single most important change. Upstream rewrote the storage/lock layer
so all the slow filesystem and S3 work runs **without blocking the event loop**.

This matters because **the fork had already solved the same problem a different
way** (see Part 2). They are two different fixes to the identical bottleneck, so
this is where almost all the manual merge work was.

- New `AsyncAGFSClient` (`openviking/pyagfs/async_client.py`) using
  `asyncio.to_thread`.
- `*_async()` variants added throughout `lock_manager.py`, `path_lock.py`,
  `lock_lease.py`, `redo_log.py`, `viking_fs.py`.
- Related follow-ups: isolate async clients per event loop (#2168), reload full
  records when updating a uri mapping (#2165), normalize AGFS error handling
  (#2227), recover stale semantic lock handoffs (#2214), semantic target-sync
  fix (#2207).

### 2. Memory & trajectories overhaul

- Removed the legacy **memory v1** format entirely; `version` field now rejects
  v1 (#2264).
- New **trajectory** model: retrieval anchors (#2255), tightened operation
  boundaries (#2221), `StoredLink` replacing `source_trajectories` for linking
  experiences (#2222).
- Skill extraction from session-commit processing (#2182).
- Searchable memory templates / `embedding_template` field (#2193, #2234).

### 3. Search & retrieval

- New lightweight **query planner** for intent analysis, configurable model
  (#2224).
- New MCP code-navigation tools: `code_outline`, `code_search`, `code_expand`
  (#2146).
- `search` tool renamed to `ov_search` to avoid an OpenClaw name clash (#2235).

### 4. Embedding input limits (relevant to the fork)

- Apply embedding input limits **in the embedder** itself (#2266) and cap queue
  input + stop retrying over-long inputs (#2197). New `ERROR_CLASS_INPUT_TOO_LARGE`
  handling so an oversized doc fails fast instead of looping.
- This is also the change that, combined with #1477-era logic, made
  `init_context_collection` **raise** `EmbeddingRebuildRequiredError` on a real
  embedding-config mismatch (the fork previously only warned — see note below).

### 5. LangChain integration

- Local **batch message writes** (#2250), recover stale OpenViking clients
  (#2246), clarified session-commit policy (#2283).
- `batch_add_messages` API renamed to `add_messages` + a `add-messages` CLI
  command (#2218, #2213).

### 6. Web Studio (the `/studio` UI)

Large front-end batch: PWA + mobile polish, OAuth-setup tab moved into the UI
(#2178, #2160, #2170), improved resource reader (#2163), request-logs touch-ups,
and bundling web-studio into pip/pipx installs so `/studio` works without Docker
(#2238).

### 7. Parsing / documents

- Offload document & PDF conversions to threads (#2237, #2220).
- Fix PDF XObject image extraction (#2199), detect binary URL imports after GET
  (#2203).

### 8. Observability

- Request-scoped HTTP profiling (`--profile`) (#2125).
- Usage/audit stored in **UTC**, bucketed per-request timezone (#2190).

### 9. Smaller fixes worth knowing

- VLM async client cache init (Codex #2208, async client cache #2168).
- `vectordb`: reject oversized byte-row strings (#2171).
- CLI: `ov wait` timeout in body not URL (#2219); report missing CLI config
  (#2279); server mode terminology `dev-implicit` → `dev` (#2260).
- Security: removed stale dependency locks (#2242).

### 10. Docs (19 commits)

Agent-integrations overhaul, MCP tool docs, OAuth setup, changelogs for
v0.3.18/0.3.19, benchmark reproduction guides. No code impact.

---

## Part 2 — What was changed by hand to merge it

`git merge upstream/main` produced **9 conflicting files**; the rest auto-merged.
The guiding rule everywhere: **take upstream's async version as the base, then
re-apply only the fork's genuinely-unique value on top.**

### Why there were conflicts at all

The fork and upstream independently fixed the *same* "storage blocks the event
loop" problem at the *same* lines of code:

| | Fork's approach | Upstream's approach (#2143) |
|---|---|---|
| How | A global 64-worker `ThreadPoolExecutor` (`run_agfs_blocking`) wrapping AGFS calls | Native `AsyncAGFSClient` with `*_async()` methods (`asyncio.to_thread`) |

Both can't coexist. **Decision: adopt upstream's native-async design, throw away
the fork's thread-pool wrapper, and re-port the two lock features the fork added
that upstream doesn't have.**

### Conflict resolutions, file by file

| File | What was done |
|---|---|
| `transaction/lock_lease.py` | Took upstream wholesale (our change was just thread-pool wrapping → discarded). |
| `transaction/lock_manager.py` | Upstream base; re-applied the fork's **lock-type capture** in both `adopt_handle` (sync) and `adopt_handle_async`. |
| `transaction/path_lock.py` | Most delicate. Upstream's `*_async` base; re-ported the fork's **lock-type cache** read-shortcut and the **transient-lock reconciliation** (`_confirm_lock_lost` / `_confirm_lock_lost_async`, 3× retry @ 50ms) into upstream's `collect_lost_owner_locks*`. |
| `transaction/redo_log.py` | Auto-merge produced **duplicate** async methods (upstream's native + our thread-pool ones calling now-deleted sync methods). Removed our duplicates, kept upstream's. |
| `storage/viking_fs.py` | Upstream base; dropped our thread-pool helpers; **re-wired the S3 Express native-rename path** (`_is_s3_express_backend()` + `_async_agfs.mv()` + `[mv]` timing instrumentation) onto upstream's async fs. |
| `storage/viking_vector_index_backend.py` | Upstream base; re-applied the fork's **parallel per-record upserts** (`asyncio.gather`) on top of upstream's async methods. |
| `utils/embedding_utils.py` | Upstream's new input-limit logic; re-added the fork's `prefetched_text` pass-through (reuse already-read file text instead of re-reading). |
| `queuefs/semantic_processor.py` | Heaviest churn on both sides. Upstream base; kept only 3 fork pieces: `prefetched_text` hand-off, the mtime change-detection heuristic, and the `_prefetched_text` return fields. **Dropped** the fork's SyncDiff perf rewrite (upstream restructured that area). |
| `models/vlm/backends/openai_vlm.py` | Took upstream's per-loop async client cache (#2168); dropped the fork's httpx keepalive workaround (no longer needed under async). |
| `tests/transaction/test_lock_manager.py` | Kept upstream's async test adaptations **and** appended the fork's new tests (`TestLockTypeCache`, `TestReconcileTransientRead`). |

### Lock features re-ported (the fork's keepers)

These survived the rebase onto upstream's async layer:

1. **Lock-type cache** — `LockHandle.lock_types` records whether each lock is a
   whole-subtree (`TREE`) or single-node (`EXACT`) lock, with a no-downgrade
   guard, so the engine avoids redundant lock reads.
2. **Transient-lock reconciliation** — before declaring another owner's lock
   "lost," re-check 3× at 50ms intervals, so a lock that's just mid-write isn't
   wrongly reclaimed.
3. **S3 Express native rename** — on Express buckets, `mv` uses the server-side
   `RenameObject` (metadata-only move) instead of copy-then-delete.
4. **Turbopuffer vector backend** — net-new, no conflicts (upstream never had it).

### One inherited behavior change (kept on purpose)

Upstream's `collection_schemas.py` now **raises `EmbeddingRebuildRequiredError`**
when a populated collection's embedding fingerprint doesn't match the current
config; the fork previously only logged a warning and continued. We **kept the
strict upstream behavior** — it refuses to silently mix incompatible vectors. See
[`testing-ingestion-s3-express-turbopuffer.md`](./testing-ingestion-s3-express-turbopuffer.md)
for how this surfaces in practice.

### Verification

- `cargo build` + `cargo test` (ragfs) green.
- `tests/transaction` and `tests/storage` green (177 storage tests pass).
- Pre-existing upstream failures identified and excluded (5 Python
  `test_collection_schemas` `_DummyEmbedder` cases + 1 Rust sqlite migration
  test — all proven byte-identical to upstream, not merge regressions).
- Live end-to-end ingestion smoke test against S3 Express + Turbopuffer passed
  (storage, vectors, async locks, native rename all confirmed).

---

## Part 3 — Post-merge follow-up (`90e77bca`)

Separately, the fork's Turbopuffer review fixes were ported in:

- Clamp query `top_k` to Turbopuffer's 10k ceiling.
- Raise (not swallow) drop/delete failures.
- True hybrid search via client-side reciprocal-rank fusion of dense + sparse.
- Fix `PathScope` depth glob semantics (`==0` exact, `<0` recursive, `≥1` bounded).

The relaxed-metadata test from that change set was **excluded** here, because it
contradicts the strict `EmbeddingRebuildRequiredError` behavior we kept (above).

---

*Branch `sync-upstream-main` holds all of the above. `trunk` is untouched and
serves as the rollback point until the final merge.*

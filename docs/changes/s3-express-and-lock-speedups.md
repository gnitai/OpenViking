# S3 Express Storage & Ingestion Speedups

This page explains, in plain terms, a set of changes that make ingestion much faster
when OpenViking stores its workspace on Amazon S3 — especially **S3 Express One Zone**.
It is aimed at operators and contributors who want to understand *what changed and why*
without reading the diffs.

## Why we made these changes

On a normal local disk, OpenViking finishes ingesting a small repository in a couple of
minutes. The same ingest against an S3 backend was taking **8–9 minutes**. Two root
causes explained almost all of the gap:

1. **Storage calls were blocking the server.** Every read/write to S3 takes 100–500 ms.
   Those calls were made directly on the server's main async loop, so while one S3 call
   was in flight, *nothing else ran* — including the OpenAI HTTP client and the background
   scheduler. The visible symptoms were OpenAI request timeouts and missed scheduler ticks.

2. **Every file move re-checked its locks against S3.** Moving a file requires holding a
   lock on its folder. The old code re-read the lock file from S3 on *every* nested
   operation to re-confirm ownership — dozens of extra round trips per move.

The changes below remove both bottlenecks and add native S3 Express support.

## What changed

### 1. S3 Express One Zone support

You can now point the workspace at an S3 Express **directory bucket**. Express keeps data
in a single Availability Zone close to compute, which is what makes it fast.

New configuration on the S3 backend:

| Field | Meaning |
|-------|---------|
| `express: true` | Use the S3 Express (directory-bucket) code path |
| `availability_zone_id` | The AZ the directory bucket lives in |
| `endpoint` | No longer required when `express: true` |

Express mode automatically rejects a few options it does not support
(`disable_batch_delete`, `directory_marker_mode: nonempty`) and warns if you point it at
an Express bucket without setting `express: true`.

### 2. Non-blocking storage calls

All storage (AGFS) calls now run on a **dedicated 64-worker background thread pool**
instead of on the main async loop. The main loop stays free to serve OpenAI requests and
run the scheduler, even while many S3 calls are in flight. This removes the OpenAI
timeouts and scheduler stalls described above.

### 3. Faster lock checks (in-memory lock-type cache)

When a write operation acquires a lock, the lock holder ("handle") now **remembers in
memory which kind of lock it holds** for each path. Nested operations under that lock no
longer re-read the lock file from S3 to re-confirm ownership — they trust the in-memory
record.

This is the single biggest ingestion speedup, but it is also the one change with a
trade-off, so it is worth stating plainly:

- **What it gives up:** the ability to notice *mid-operation* that a lock it already holds
  was taken away by someone else.
- **Why that is safe in the supported deployment:** OpenViking runs as a **single replica**
  (one process), enforced by a process-exclusive lock on the data directory. No other
  process exists to take a lock away, so the cached answer is always correct. The lock
  holder also refreshes its locks every `lock_expire / 2` seconds (150 s by default) so a
  lock it is actively using does not expire underneath it.
- **When it would NOT be safe:** running two or more writer processes against the **same**
  workspace at the same time. That configuration is not supported (see *Limitations*).

### 4. Native server-side rename on S3 Express

Moving a file used to mean *copy then delete* — two operations, with the data copied byte
for byte. On S3 Express, a move now uses the bucket's native **rename** operation: the
object is moved server-side in one atomic step, with no data copy. Moves are dramatically
faster and cannot leave a half-copied object behind.

### 5. Concurrent sidecar writes

Each folder gets two generated summary files — `.overview.md` and `.abstract.md`. They
used to be written one after the other, so their S3 latencies stacked up. They are now
written **in parallel**. A `write_concurrency` limit (default 4) caps how many folders
flush their sidecars at once, so we get the speedup without flooding S3 Express with
requests.

### 6. Transient-lock reconciliation

A brief, recoverable S3 read error looks identical to "the lock is gone." Previously a
single such glitch could make a handle wrongly believe it had lost its lock and drop it.
Now, before declaring a lock lost, the code **re-checks a few times** (3 attempts,
50 ms apart). A genuinely different owner is detected immediately; a momentary blip
recovers. This makes ingestion more robust under S3 throttling.

### 7. Smaller wins

- **Parallel index updates:** vector-store records for a moved tree are updated in parallel
  instead of one at a time.
- **Reused file text:** text read during the summary step is passed forward to the
  embedding step so the same file is not fetched from S3 twice.
- **Per-phase timing logs:** moves now log a breakdown (lock, copy, vector update, remove)
  to make future performance work easier.

## Configuration reference

| Setting | Default | Notes |
|---------|---------|-------|
| `express` | `false` | Enable S3 Express directory-bucket mode |
| `availability_zone_id` | — | Required when `express: true` |
| `write_concurrency` | `4` | Max folders flushing sidecars at once |
| `lock_expire` | `300` s | A lock is considered stale after this; refresh runs at half this interval |

## Limitations

These speedups are designed for OpenViking's **single-replica** deployment model and do not
change it:

- **One writer per workspace.** The local vector index (RocksDB) and task queue (SQLite),
  the in-process lock manager, and the lock-type cache (item 3) all assume a single writer.
  Run **one** server replica per workspace; scale by giving that replica more CPU/memory,
  not by adding replicas.
- **S3 Express One Zone is single-AZ.** It is fast because it lives in one Availability
  Zone, but that also means it is **not** multi-AZ durable. AGFS is OpenViking's source of
  truth, and lost source data is unrecoverable, so do not treat a single Express bucket as
  the only copy of important data — replicate or back it up to a multi-AZ (Standard S3)
  bucket.

See also: [Path Locks and Crash Recovery](../en/concepts/09-transaction.md) and
[Storage Architecture](../en/concepts/05-storage.md).

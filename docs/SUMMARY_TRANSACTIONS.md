# Experimental isolated summary transactions

This document describes the legacy protocol 3 serial transaction path.
The v7 priority integration uses the separate protocol 5 contract documented in
[PRIORITY_SCHEDULER_LOCAL.md](PRIORITY_SCHEDULER_LOCAL.md).

This feature is opt-in and must remain unqualified until tested on the exact model,
backend, binary, runtime settings and maximum main-session/cache sizes in use.
It serializes summary work through the same engine and device session. It is not
an independent concurrent KV slot and does not copy the model weights.

## Protocol version 3

- `SUMMARY_CAPS` returns `SUMMARY_CAPS 3 16384` for a supported configuration,
  or version `0` when unavailable. This is protocol support, not qualification.
- `SUMMARY_BEGIN <max_main_tokens> <max_snapshot_bytes> <max_protected_bytes> <path>`
  requires three positive integer limits, followed by the private snapshot path.
  A successful response is `SUMMARY_HELD <tokens> <bytes> <ms>`.
- The summary uses text `GEN`. Prompt tokens plus output budget plus eight tokens
  must not exceed 16,384. Checkpoint creation and parking are disabled.
- `SUMMARY_END <same path>` returns `SUMMARY_RELEASED <tokens> <bytes> <ms>` only
  after restoring the saved device state and original protected caches.
- Session-file `SESSION`, `SWAIT`, `SERR` and `FATAL` responses retain their meaning.
  Every BEGIN `SERR` is before live/cache mutation, including `published=1`, which
  means the snapshot file was published, not that ownership was transferred.
  Uncertain BEGIN outcomes and any END failure require a poisoned inference lane.

Version 3 also supports explicitly pristine EMPTY transactions. Warm DISK mode has
strictly positive HELD/RELEASED token and byte counts. EMPTY has exactly zero for
both counts. The caller must validate matching BEGIN/END counts.

A startup provenance marker, rather than `live_ok=false` alone, authorizes EMPTY.
Any ordinary generation or RESTORE attempt clears it before parsing or prefill,
including malformed and cancelled requests. Live tokens, checkpoints, parked/reuse
cache, image ownership or poisoned state also prohibit EMPTY. Only successful
EMPTY rollback restores the marker, supporting repeated initial compaction batches.

EMPTY neither writes nor removes the supplied path: it is an exact-match transaction
identity. A pre-created zero-byte placeholder is not a KV snapshot. END quiesces the
device, performs the existing canonical sequence reset, checks device completion,
restores original vision/metadata/cache ownership, and exposes logical empty state.
Unused GPU bytes need not be identical. Acceptance requires deterministic subsequent
foreground output and identical visible/cache state, not unused-memory byte equality.

Uncertain BEGIN still poisons inference. If mode is unknown, do not claim a usable
snapshot exists. Known EMPTY has no saved main KV image to recover; its reset/upload
failure also poisons the lane. No dummy foreground request is used to create a cache.

Version 2 has rollback envelopes but cannot summarize before the first main request.
Version 1 does not enforce the rollback admission or qualification envelopes.
Do not enable this feature against a version 1 engine.

## Memory admission and qualification

SAVE streams KV to disk. END currently reads the complete snapshot into host RAM.
Checking SAVE memory alone is insufficient. Before ownership moves or summary
execution can modify the main device state, version 2 checks:

1. Actual main tokens against the qualified token limit.
2. Actual written snapshot bytes and protected-state bytes against their limits.
3. Fresh available host memory against `session_read_peak_bytes(snapshot_bytes)`
   plus a 1 GiB summary-scratch margin and the existing free-memory floor.

Protected-state accounting includes the full checkpoint chain, parked/reuse cache,
original image embedding and pointer storage, saved M-RoPE positions, plus a 16 MiB
metadata allowance. These allocations are already resident and reflected in the
available-memory measurement; they are not subtracted twice. The separate bound
prevents a short empty-cache test from qualifying a large cache configuration.

The 1 GiB scratch margin and metadata allowance are conservative engineering
estimates, **not measured peaks, allocator bounds or memory reservations**. Prefill
may allocate additional buffers, and summary-side state can remain allocated while
END reads the snapshot. Qualification must measure the simultaneous high-water
mark, including a full 16K summary prompt/output budget, retained main/cache/vision
state and END. If extra allocation exceeds the margin, do not qualify this build;
increase the budget or revise the design, then rebuild and repeat qualification.
END rechecks memory because other processes and changing memory pressure can still
make restoration fail. This feature does not make OOM impossible.

The Python configuration must supply explicit positive values for
`atomic_summary_qualified_max_main_tokens`,
`atomic_summary_qualified_max_snapshot_bytes`, and
`atomic_summary_qualified_max_protected_bytes`, in addition to the qualification
flag and private snapshot path. Use only experimentally established limits for the
exact deployment. A passing short test does not qualify a 64K main context. Do not
silently lower the user's context or raise OS memory limits to pass a test.

## Preservation and failure behavior

BEGIN never reclaims original caches to create room. Original checkpoint buffers,
whole parked-cache bookkeeping, clock/tail state and vision embeddings are retained.
M-RoPE host/device positions and the embedding pointer ownership are restored before
release. A failed device restore or M-RoPE upload terminates the engine. The frontend
must STOP and drain a cancelled summary before END and must keep foreground requests
blocked until successful release. Retain the snapshot on uncertain/fatal failure.

Protected caches are process-local, not crash-durable. QUIT/EOF/crash stops the
process; the disk image survives, but it contains only the existing session format's
selected checkpoint, not the entire protected checkpoint chain.

Bounded streaming END is not implemented. It would require bounded first-pass file
validation/indexing, a second pass over the same protected file handle, chunked
transfers with checked array geometry/offsets, and fatal-on-partial-write handling.
That is a separate parser/transfer change requiring corruption, truncation,
concurrent-file-change, transfer-failure and GPU parity tests.

Cold-state qualification must include repeated initial batches, cancellation, wrong
END path, unknown BEGIN and failed END, with vision enabled, followed by deterministic
real foreground comparisons. Cancelled or failed ordinary prefill must not regain
EMPTY eligibility merely because its token/checkpoint vectors are empty.

## No-MTP disk-session regression

Disk SAVE/read-limit helpers accept a nullable draft state, matching ordinary parked
snapshot validation/restoration. A no-MTP image has exactly the main QSA layers;
a draft-enabled image has those layers plus one draft layer. Existing reference
helper overloads still include the draft and preserve the MTP file layout. The
server's SAVE and RESTORE source/limits/validation/transfer calls consistently pass
`use_mtp ? &mtp.kv_state() : nullptr`; a dummy uninitialized draft is not a real layer.

From the configured Metal build, explicitly build `summary_disk_cpu_test`, then
run that binary. It uses production snapshot/file/read-limit/restore functions with
host transfer mocks, not Metal or a model. It reproduces the old dummy-draft extent
error, roundtrips actual nonempty no-MTP and MTP files, checks restored state and
wrong-draft rejection, and verifies byte-identical MTP reference/pointer files.

An optional `summary_disk_gpu_test` build target uses the same tiny resident INT8
fixture and real Metal buffer allocation/copies, without any model or downloads.
It is not run by default. Only the designated GPU operator may run it. It exercises
generic checkpoint transfers, not the production Metal Window checkpoint hooks;
it cannot replace the actual IQ3 warm SAVE/summary/RESTORE qualification. Device
fixture payload is below 1 MiB; host file buffers add tens of MiB plus Metal runtime
overhead. Both targets remain outside the default build.

# Experimental Metal summary state bank (protocol 4)

This local implementation is not inference-qualified. It shares one loaded model and
serial compute; it does not run foreground and summary GPU work concurrently. Protocol 3
and its qualified disk rollback envelopes remain separate. No automatic disk fallback is
permitted when protocol 4 is requested.

## Wire contract

- `SUMMARY_STATE_CAPS` replies `SUMMARY_STATE_CAPS 4 16384`, or version 0 when unsupported.
- `SUMMARY_STATE_BEGIN <max_slot_bytes> <txid>` publishes only after admission and allocation.
- `SUMMARY_STATE_HELD <txid> <main_tokens> <allocated_bytes> <epoch>` acknowledges publication.
- `SUMMARY_STATE_END <txid>` returns the saved main bank without a new state allocation.
- `SUMMARY_STATE_RELEASED <txid> <main_tokens> <allocated_bytes> <epoch>` repeats the HELD tuple.
- `SUMMARY_STATE_REFUSED <txid-or-INVALID> <reason>` is a clean, non-published BEGIN refusal.
  Wrong END identities refuse while leaving the active bank sealed. Unknown transition failures
  are FATAL, not a claim that main can safely run.

The budget is canonical positive decimal, at most 9007199254740991. Transaction IDs have
1–64 ASCII alphanumeric characters. Epoch is a positive process-local monotonically
increasing uint64. While held, only bounded text GEN, END, CAPS, STOP and QUIT are permitted.
Input + requested output + 8 must be at most 16384. Checkpoint creation and parking are off
for the summary lane. The Python scheduler owns FIFO, client cancellation, STOP/drain, and
foreground priority. END must complete before foreground work resumes.

## Ownership and admission

The Metal Window exchanges its recurrent state, both convolution histories, INT8 KV/scales,
indexer pooled/tail/dead buffers, PLE history, host PLE history, normalized rows and committed
window bookkeeping with an independently allocated 16384-cell bank. The main sequence is
not serialized, copied to a file, or cleared. Main checkpoints and the entire conversation
cache move aside with their ownership and LRU bookkeeping intact. Main token/image vectors,
embedding pointers, M-RoPE globals, session descriptors and PLE token history are restored.
The original QSA descriptors are not rebound to the summary buffers: separate descriptors
prevent a 16K min-capacity binding from shrinking the main session.

Before switching, outstanding host prompt work is joined and discarded and pending main
GDN folds are completed using the main scratch. Merely retaining a deferred-fold count
would be unsafe because summary work reuses its inputs. END discards the summary's deferred
fold, restores main, clears capacity-specific dispatch caches and marks shared selection
scratch for regeneration. Shared prompt scratch always has permanent foreground capacity;
its size does not become 16K merely because the summary bank is active.

Support is restricted to the current Metal 48-layer/12-QSA configuration, including warmed expert streaming,
without MTP, DFlash2, taps, batch slots, layer split, peers or multiple GPUs. Pending-logit
diagnostics and destructive profiling/skip modes are also refused. The ordinary
foreground context remains unchanged. Protocol 4 has no protocol-3 main-token/checkpoint/
parked-cache envelope, because those allocations remain resident and unmodified. This is
an implementation boundary, not evidence of 64K workload qualification.

Admission requires known normal macOS pressure and stable swap, available RAM for the bank
plus 1024 MiB scratch allowance plus the configured free-memory floor, and a caller byte
budget covering all accounted bank storage. It checks again after allocation before the
switch. The second check is deliberately conservative: it still includes bank bytes although
those allocations are now resident. Metal bank allocation is nonfatal and cleans up partial
allocation on refusal. Accounted bytes include requested Metal buffer bytes and bank host
storage, not the OS's page-rounded physical footprint. No OS memory limits are changed.

Streaming retains one weight-only expert arena, slot map and LRU. A bank switch requires
an already initialized arena and, for matrix prefill, an already allocated `pg_` scratch
layout. The non-streaming `pb_` capacity is not a streaming warmup condition. Bank switches
cancel wholly queued prefetches and wait at most 60 seconds for running expert reads, after
the GPU's synchronous layer handshakes finish. BEGIN refuses before state publication if
the prompt region is still held or the drain times out; END failure is FATAL/poisoned.
Prompt-region ownership is explicit and scope-released only after a checked GPU fence.
Shared expert-weight cache hit counts can change; this is distinct from preserved main
conversation checkpoints and parked-cache ownership. No arena resize, new cold arena or
prompt scratch initialization happens after bank publication.

Cold start is **not supported** by the default prompt-chain path. Existing shared prompt
scratch must have capacity at least the configured prompt chunk (default 2048, clamped
64–8192), populated by an earlier foreground request. This avoids potentially multi-GiB
cold scratch allocation after publication. Explicit window-only prefill already uses the
allocated decode scratch, but streaming still requires its foreground-initialized arena. Public capability should conservatively say cold_start=false and
requires_foreground_warmup=true. No Dstar cold-start parity is claimed.

## Validation status

The standalone CPU parser/gate test is `tests/summary_state_bank_test.cpp` (53 checks).
It validates grammar, bounds and protocol gating, not GPU isolation or output parity.
No model load, real inference, long stress run or recovery inference was performed for this
implementation. User testing is required. In particular, image/main-cache preservation,
cancel/preemption, pressure/OOM behavior and numerical output parity are not measured here.

## Bound clone capacity (personal integration)

`SUMMARY_CLONE_CAPS` may return `SUMMARY_CLONE_CAPS 1 <foreground_context> <physical_cells>`.
The optional last field is the native window's actual allocation, including explicitly
reserved clone headroom and alignment. Legacy three-field replies retain the foreground
limit. Clients validate the reported bounds; they must not infer additional space from an
environment variable. This does not change the separate 16K fresh-summary bank described
above or expand normal foreground requests.

Cached clone input and output, and serial disk-preserved emergency continuation, are
admitted against that physical capacity. Metal KV scratch and the host/device image
position tables use the same checked capacity calculation. The position tables must not
remain at the logical foreground size when a clone crosses that boundary. Absolute
position limits and rollback ownership checks remain enforced. If physical space is also
exhausted, recovery still retains originals and may use independent extraction. Earlier
checkpoint cloning is not implemented by this capacity fix.

The personal 64K M1 Max integration was checked with an independent copy of an existing
64,843-token session: all 64,843 tokens were reported as cached, 3,580 additional input
tokens were read, and a short validation response completed beyond the logical 65,536-token
boundary within the reported 73,856 cells. The main owner and original session file were
preserved. This is a targeted recovery-path check, not a general summary-quality claim.


## Earlier-checkpoint recovery (personal Metal opt-in)

`atomic_summary_checkpoint_enabled:true`, native INFO `summary_checkpoint=1`, and
`STRATA_ATOMIC_SUMMARY_CHECKPOINTS=1` enable earlier-state recovery. The app sends
an exact source prefix as CPU-only metadata; it never joins the model prompt.
`SUMMARY_CHECKPOINTS <main_count> <main_hash>` lists only committed checkpoints of
that exact live prefix. `SUMMARY_STATE_CHECKPOINT <budget> <id> <main_count>
<main_hash> <checkpoint_count> <checkpoint_hash>` validates both bindings, clones KV
into the separate bank and restores that checkpoint's recurrent/convolution, PLE
and indexer state. The private suffix position table starts at that checkpoint's
next absolute position. The full main count remains the rollback identity.

Only checkpoints covering the entire archived source qualify; no missing image
rows or partial source gaps are replayed. Expanded image placeholders require the
completed-history receipt. An unavailable or stale checkpoint retains originals
and the existing fallback. A checkpoint-selection margin of 4096 output tokens is
not an output cap: generation uses all actual remaining physical room.

The personal configuration uses existing `--prompt-cache-spread` and
`--prompt-cache-every 4096` retention. Opt-in SAVE retains up to three exact-prefix
anchors and admits the peak for all copies before allocation. Other configurations
still save one checkpoint; the session format is unchanged.

Validation on the personal M1 Max / IQ3_XXS: a synthetic 11,782-token foreground
session restored three saved checkpoints. Two successive server restarts each
selected and reused the same 1,255-token checkpoint, completed a short JSON response
with 11,074 new instruction tokens, and left the original foreground session file
unchanged during summary. A separate copy of the existing 64,843-token user cache
also restored, reused all 64,843 tokens, accepted 3,580 new input tokens and released
the lane successfully. The production manifest and session remained unchanged.
CPU parser tests passed 95 checks and session-file tests 209 checks. App verification
passed 8,697 web tests, 525 extension tests and platform Rust suites; the focused
adapter tests passed 41 tests. These are targeted path/isolation checks, not a claim
of long-summary quality, GPU numerical parity, OOM stress coverage or non-Metal support.

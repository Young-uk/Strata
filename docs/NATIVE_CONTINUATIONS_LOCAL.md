# Experimental bounded native continuations

Experimental implementation deployed in the personal v7 integration. The checks below
do not establish real-model qualification. This does not
claim output parity, a latency improvement, or full legacy GEN feature coverage.

## Protocol 1

All commands use the existing single engine stdin/stdout connection. The caller
must serialize commands and consume each terminal response before issuing another.
An `ANEXT` is real compute on the existing model and verifier, not a repeated GEN.

- `ASYNC_CAPS` returns `ASYNC_CAPS 1 512 1 text` only for the supported configuration.
  A zero version means unavailable. Require the exact advertised tuple.
- `AGEN <id> <main|summary> <max_new> [key=value ...] <token,token,...>` admits a
  continuation and returns `AREADY <id> <committed_inputs> <prompt_tokens>`.
  IDs are 1 to 64 ASCII letters/digits. Max output is positive.
- Supported keys: temperature, top_p, top_k (1..64), min_p, penalty_last_n (0..4096),
  penalty_repeat, penalty_freq, penalty_present, seed. Unknown keys are refused.
- `ANEXT <id>` runs at most 512 foreground prompt inputs, exactly one remaining
  summary prompt input, OR one decode input, using the existing verifier windows. Prefill never samples. Decode emits `AT <id> <token>`.
  A nonterminal step ends with `AYIELD <id> <prefill|decode> <committed> <produced>`.
  A terminal decode ends with `ADONE <id> <stop|length> <produced>`.
- `ACANCEL <id>` drops only the named record, answering `ACANCELLED <id>`. It
  does not cancel the other lane or rewind either bank. The exact committed prefix
  remains eligible as live state. A completed ID may be cancelled to release it.
- Admission and ownership refusals are `ASYNC_REFUSED <id|INVALID> <code> <reason>`.
  Native compute/transition failures are `FATAL ...`, followed by process exit;
  their state must not be treated as safely resumable.

Use existing `SUMMARY_STATE_BEGIN <budget> <transaction_id>` before admitting a
summary lane. BEGIN is allowed between main quanta. Each ANEXT selects its own
bank. END refuses while the summary continuation is active. After summary
completion or explicit cancellation, `SUMMARY_STATE_END <transaction_id>` returns
to the latest main state, including any foreground progress made since BEGIN.
Never use STOP, END, or ACANCEL to implement scheduler yields.

## Ownership and preserved state

Each lane retains its original prompt length, full prefix, committed-input cursor,
output count, output limit, and sampler settings/seed. The final emitted output
is pending at ids[cursor], not yet part of the committed state. The existing
verifier derives Philox's counter from that absolute cursor. Penalties use the
original prompt boundary, not a new slice boundary. Resumption does not repeat
cache matching, reset state, rerun a committed prefix, or resample an output.

Metal switches the session, KV/recurrent state and embedding/M-RoPE pointers only
at a completed window boundary. Host live tokens, image storage/pointers, flags,
checkpoint objects, parked conversation cache, K/V floors, segment plan and cache
bookkeeping swap with it. Shared sampler scratch is rebound on every ANEXT.
A summary quantum is limited to one input even during prefill. This prevents a
512-input background prefill quantum from outweighing foreground decode turns.
There is only one model and no GPU overlap. This supplies compute quanta; weighted
foreground/summary scheduling is the server's responsibility.

## Explicit limits

Only the supported Metal summary-bank configuration is admitted: no batch,
multi-GPU, speculative drafter, pipeline, elastic KV, asynchronous adaptive worker,
or output-head aliasing. The existing native bank also requires warmed state.
The summary lane has 16,384 positions including an eight-position safety margin.
The main lane keeps its configured context size.

AGEN admission is text-only and accepts empty main state or an exact live prefix.
A nonmatching main context is refused instead of discarding checkpoints or parked
conversations. Use the separate APREP capability below for full foreground
admission, including images and cache matching. A caller must report unsupported
options and must not silently downgrade or drop them. Ordinary GEN is blocked
while either continuation is active. APREP adds conditional capture hooks to the
legacy GEN body; ordinary GEN does not create or emit a continuation.

## Checks run

- Clang C++20 Metal syntax check of generate.cpp: passed; one existing override warning.
- CPU native_continuation_test: passed (parser, ownership/cursor, pending output).
- CPU summary_state_bank_test: 53 checks passed.

No model was loaded. No generation, conversation, GPU performance or numerical
parity tests were run. CUDA, HIP and SYCL builds were not run.

## Foreground legacy admission: APREP 1

This second, separately negotiated capability extends the experimental implementation.
`APREP_CAPS` returns `APREP_CAPS 1 legacy1`, or version 0 if unavailable.

`APREP <id> <max_total_output> <GEN|GENI> <legacy options/path/token list>` runs
exactly one initial legacy GEN/GENI output on the main bank. Do not include an
additional max_new after GEN/GENI. The initial prefill, image preparation,
checkpoint matching, conversation parking/restoring and prompt checkpoint creation
use the existing path. Main image embeddings and M-RoPE globals then stay with its
bank throughout subsequent weighted decode. The existing warm-configuration
restriction still applies to capability negotiation.

The caller must hold exclusive protocol ownership through all legacy PP/T/LP/DONE
lines until `APREPARED <id> <committed_inputs> <produced> <status>`. The status is
active, stop, length, cancel, or failed. An active result has emitted exactly one
output; its committed input prefix plus that pending output and actual sampler
seed, bias, penalty origin, cache policy and logprob options are captured. Continue
with ANEXT. stop/length means no further generation is needed. failed creates no
new continuation, and may follow an ordinary legacy ERR. A malformed APREP header
instead returns ASYNC_REFUSED without entering the legacy path.

Supported APREP keys are the sampling/penalty keys above, pen_from, logit_bias,
cvec, logprobs, allowed, ckpt, pin, and img_keys. Other keys are refused before
legacy execution. ANEXT emits `ALP <id> <existing LP payload>` immediately after tagged AT when
requested; these lines remain owned by that ANEXT until AYIELD/ADONE.

APREP prioritizes foreground prefill and the first token exclusively: summary
receives no compute during that initial step. Weighted interleaving resumes for
subsequent foreground decode. This is not a claim of bounded foreground-prefill
latency or numerical equivalence to a different prefill kernel. Cancellation
while APREP is running may use STOP for that exclusive owner; always drain through
APREPARED before selecting another lane. A cancellation arriving after first-token
completion may see active and must then cancel that matching continuation.

The scope-exit APREPARED boundary also covers ordinary early returns/continues.
A FATAL or process death still invalidates the process and is not a resumable
admission refusal. No live-model, image conversation, cancellation-race or numerical
parity test has been run for this extension. Local build artifacts and rollback backups are not included in this branch.

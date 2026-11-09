# Staged weighted summary scheduler

This source is an isolated development copy. It is not installed in Atomic Chat.
The existing v6 runtime, application, model files and user data remain unchanged.

## Implemented behavior

The native continuation protocol retains each lane's committed input cursor,
pending output, original penalty boundary, sampler seed/settings and token prefix.
It does not stop or resubmit a summary when foreground work arrives. A dedicated
Metal bank switch retains both contexts and flushes outgoing deferred work before
shared scratch changes. One native compute turn owns the GPU and protocol stream.

The Python scheduler admits one foreground request and one summary, retaining their
owners until completion or acknowledged cancellation. When both are ready, eight
foreground turns precede one summary turn. Background prefill advances one input
per turn, as does decode. Foreground initial preparation uses the original GEN/GENI
cache/image path through its first token, exclusively; weighted interleaving starts
for its remaining decode. The direct AGEN protocol also supports bounded text
prefill, but the server uses APREP for full foreground preparation. The ratio
is a scheduling policy, not an 8:1 measured time or throughput guarantee. A slow
output consumer cannot retain the compute permit between acknowledged quanta.

Only the owner's cancellation, disconnect, explicit suspension or shutdown cancels
a summary. Consumer close during a native heartbeat drains the owned quantum before
cancellation and permit release. Unknown native identity, process replacement,
malformed progress or uncertain bank/cancellation outcomes poison admission. Already
admitted requests are blocked too. There is no automatic recovery inference.

Request progress and counters are isolated. The latest main token count is allowed
to grow during a summary transaction; its immutable transaction/epoch/allocation
identity must still match. Timing fields are unavailable rather than invented.
Original messages and archive commit behavior are not changed by this backend.

## Opt-in and capability binding

Keep `atomic_summary_mode` as `state-bank-v1`, and explicitly set
`atomic_summary_scheduler` to `weighted-quanta-v1`. Optional
`atomic_summary_foreground_weight` defaults to 8, in 1..64. Existing strict artifact
and launch-configuration pins remain required and now include the scheduler modules
and settings. Discovery requires the exact native `ASYNC_CAPS 1 512 1 text` tuple.
Foreground also requires `APREP_CAPS 1 legacy1`. Public summary protocol is 5 with
isolation `shared-weights-priority-v1`, `foreground_preparation: legacy-first-token-v1`,
`foreground_prefill_exclusive: true`, and `continuation_text_only: false`. The staged
app patch validates this exact contract; an old app that understands only protocol
4 will not enable it. No existing
configuration or qualification receipt has been rewritten to enable it.

## Release blockers and explicit limits

The direct AGEN path remains restricted to text and exact live prefixes. The server
uses the added APREP foreground bridge instead: the original prompt/image/cache
matching and parked-conversation restoration run once through the first output.
Its actual sampler, bias, penalty boundary, cache policy and pending output then
continue without reset or replay. Initial foreground preparation is exclusive;
summary is not guaranteed compute during a long foreground prefill, only during
subsequent weighted decode. This is not fully preemptible foreground prefill.

SAVE/RESTORE, elastic VRAM changes, elastic image encoding, automatic reload and
idle unload are explicitly unavailable in priority mode. Non-elastic image encoding
uses the same FIFO as native compute. Application-level drain and process shutdown
remain the supported stop path. A process change requires a new runtime.
Full-response decode timings and expert-cache counters remain unavailable rather
than incorrectly reusing first-token measurements. Prompt timing, committed token
progress and prompt-cache reuse are retained.
Streaming or explicit Metal prompt-window mode is required; native BEGIN still
checks actual shared scratch/arena warmup and memory. Capability is not readiness.

These limitations must accompany any experimental packaging decision. No installed
app, runtime configuration, qualification pin or user data has been changed. No
claim of full Dstar parity or empirical safety follows from these checks.

## Validation

The Metal release target compiles and links on the connected Mac, using two compiler
jobs. Focused CPU/fake-protocol checks cover weighted selection, request isolation,
cancellation, disconnect drain, stale identity, process replacement, changed main
token counts, pre-BEGIN cancellation, uncertain transition poisoning and control
rejection. Existing snapshot/state-bank/qualification checks are also run. The latest combined
Python CPU/fake-protocol suite passes 141 tests. The staged app passes 167 mocked
checks, its TypeScript check and scoped ESLint. A concurrent fake-engine check
observes summary output before foreground completion without a priority STOP.

No model was loaded or queried, and no real conversation, long-context, image,
performance, GPU state-parity or stress test was performed. Such behavior remains
unverified and the user retains responsibility for requested real-model testing.

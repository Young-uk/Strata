# Conditional summary admission (local bounded release)

This is a safety restriction on the opt-in serial summary transaction. It is not
qualification of 64K rollback or a reservation of RAM. Foreground context stays
65536. No OS memory limits, cache configuration, or model settings are changed.

## Contract

CAPS remains protocol 3. BEGIN still requires explicit qualified main-token,
snapshot-byte and protected-byte limits. Configuration capability does not prove
that a particular foreground state or summary workload has been qualified.

Before any snapshot capture, BEGIN requires actual foreground state with no
image markers, embedding buffers/pointers, or nonidentity vision positions; at
most six main checkpoints and four parked entries (never exceeding configured
slots). Every main/parked checkpoint and parked live state must be image-free,
single-stage, and within the qualified main-token bound; each parked chain is
also limited to six checkpoints. Retained reusable KV requires explicit text
provenance and bounded captured extent, not merely its remaining unchanged
prefix. Unknown provenance fails closed. The ordinary
vision option and multi-checkpoint cache configuration can remain enabled. A
request in an unsupported state is refused without eviction or truncation.
The frontend maps a structured BEGIN refusal to HTTP 409 and preserves the main
conversation. Byte envelopes remain independently enforced.

On macOS only, BEGIN reads `kern.memorystatus_vm_pressure_level`, `vm.swapusage`,
and HOST_VM_INFO64 swapouts. Missing telemetry, non-normal pressure (anything
other than 1), increased swap-used bytes, or changed cumulative swapout count
refuses optional summary work. Two samples 100 ms apart are checked before SAVE.
A further sample is compared with that baseline immediately before ownership
transfer, after SAVE (or on EMPTY admission). Stable preexisting swap is allowed;
falling swap usage does not hide an increase in swapout count. These are host-wide
signals, so another application's activity can conservatively defer a summary.
They do not predict future pressure or reserve memory.

The existing admission for complete rollback read peak plus 1 GiB scratch and the
configured free-memory floor remains required before BEGIN. Protected state has
its own byte envelope and is not charged twice against available RAM.

END samples pressure/swap again. Deterioration or unknown health is logged, not
used to prohibit recovery. DISK END still requires its existing available-RAM
allocation admission before reading the snapshot, validates it before device
writes, and keeps a failed restore sealed and recoverable. EMPTY END performs its
existing checked reset without allocating a disk image. A transfer failure stays
fatal/poisoned; no failure is reinterpreted as a successful restore. Pressure
alone must not make safe recovery impossible. No shape restriction is applied
to END, so a snapshot already held under an earlier capability can be restored.

## Validation and limits

CPU tests cover six/seven checkpoint boundaries, deep image markers, unknown
reuse provenance, retained captured extents, total protected-byte refusal, unknown and abnormal pressure, swap growth/churn/counter regression,
and unchanged transaction sealing/ownership behavior. A real macOS telemetry
probe compiled and returned valid pressure/swap counters without GPU use.

These guards do not qualify the maximum 16K combined summary context or the
2048-token frontend output cap. That needs paired live-model tests of the final
engine binary, including cancellation and restore parity at the selected main
state and byte bounds. A short main-session test does not qualify a 64K main
snapshot. Keep qualification limits at measured values until those tests pass.
Non-macOS platforms retain their existing memory admission; this change does not
claim pressure/swap protection or backend validation there.

## Private cache candidate

The expanded text-cache shape is a private, unqualified candidate until paired
live-model tests pass. A 1 GiB protected-byte envelope is a combined limit, not
permission to retain 1 GiB of parked cache plus six main checkpoints. The
configured cache budget excludes main checkpoints; the summary envelope counts
both. Shapes exceeding any token, snapshot, protected-byte or count bound are
deferred without eviction. Foreground cache settings stay unchanged.

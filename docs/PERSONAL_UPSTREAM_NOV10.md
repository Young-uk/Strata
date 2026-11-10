# Personal upstream integration, 2026-11-10

The personal branch integrates upstream `504b14334` while preserving its
summary, cache ownership and scheduling extensions. The geometry allocation
conflict uses the upstream geometry constants and retains the personal
`4 * MAXC` scratch capacity. The native build is qualified independently;
no previous-build disk-session compatibility is assumed.

The original full server run exposed five tests that constructed a Service
without its personal summary lane, and upstream test assumptions reproduced
on unchanged upstream on macOS: CUDA/HIP policy tests used the host platform,
and the decoding fixture replaced subprocess.Popen globally, intercepting
lifeline's ps probe. Fixtures now explicitly provide their intended environment.
Mac policy remains covered separately. The MCP startup failure also exposed a
race: an already-exited process could be reported before stderr was collected.
The request path now waits for the same bounded stderr collection as the EOF
path, preserving the startup diagnostic.

Validation uses fake engines and small CPU/Metal fixtures. No real model
inference or throughput measurement is performed.

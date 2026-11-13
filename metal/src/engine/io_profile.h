// Diagnostic counters only. Worker time overlaps across threads and GPU work.
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::metal_engine {
inline int metal_io_profile_mode() {
    static const int mode = [] { const char* p = std::getenv("STRATA_METAL_IO_PROFILE");
        return p && p[0] == '1' && !p[1] ? 1 : p && p[0] == '2' && !p[1] ? 2 : 0; }();
    return mode;
}
struct IoCounters {
    uint64_t bytes = 0, experts = 0, worker_us = 0, hits = 0, pre_hits = 0, misses = 0;
    double stall_ms = 0, callback_ms = 0;
};
inline IoCounters io_delta(const IoCounters& a, const IoCounters& b) {
    auto sub = [](uint64_t x, uint64_t y) { return x >= y ? x - y : 0; };
    return {sub(b.bytes,a.bytes), sub(b.experts,a.experts), sub(b.worker_us,a.worker_us),
            sub(b.hits,a.hits), sub(b.pre_hits,a.pre_hits), sub(b.misses,a.misses),
            std::max(0.0,b.stall_ms-a.stall_ms), std::max(0.0,b.callback_ms-a.callback_ms)};
}
struct IoProfile {
    IoCounters total;
    uint64_t windows = 0, rows = 0;
    double gpu_span_ms = 0;
    void add(const IoCounters& d, int n, double span) {
        ++windows; rows += n; gpu_span_ms += span;
        total.bytes += d.bytes; total.experts += d.experts; total.worker_us += d.worker_us;
        total.hits += d.hits; total.pre_hits += d.pre_hits; total.misses += d.misses;
        total.stall_ms += d.stall_ms; total.callback_ms += d.callback_ms;
    }
    void report(std::FILE* f, bool summary, bool prompt) const {
        std::fprintf(f, "strata metal io: lane=%s phase=%s windows=%llu rows=%llu gpu_span_ms=%.3f "
            "host_callback_ms=%.3f expert_wait_ms=%.3f io_worker_ms=%.3f completed_bytes=%llu "
            "completed_experts=%llu hits=%llu prefetch_hits=%llu misses=%llu; cumulative; "
            "GPU span includes host turns; worker time overlaps; completed I/O can cross windows\n",
            summary ? "summary" : "foreground", prompt ? "prompt" : "window",
            (unsigned long long)windows, (unsigned long long)rows, gpu_span_ms,
            total.callback_ms,total.stall_ms,total.worker_us/1000.0,
            (unsigned long long)total.bytes,(unsigned long long)total.experts,
            (unsigned long long)total.hits,(unsigned long long)total.pre_hits,(unsigned long long)total.misses);
    }
};
} // namespace strata::metal_engine

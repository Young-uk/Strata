// Host-side Metal diagnostics. No GPU commands or numerical state are changed.
#pragma once
#include "strata/core/progress.hpp"
#include <cstdlib>

namespace strata::core {
// Phase scopes are owned by the Qwen submission thread; the watchdog only reads them.
struct MetalActivity {
    std::atomic<const char*> phase{"idle"};
    std::atomic<int64_t> phase_since_ms{0};
    std::atomic<uint64_t> segments{0}, commands{0}, dispatches{0};
    // Wall time on the calling CPU thread; GPU waits include host/event scheduling.
    std::atomic<uint64_t> encode_us{0}, wait_us{0}, callback_us{0}, prepare_us{0}, ple_us{0}, compile_us{0};
};
inline MetalActivity& metal_activity() { static MetalActivity a; return a; }
inline bool metal_activity_timing();
inline void metal_activity_report(std::FILE* f) {
    const auto& a = metal_activity();
    std::fprintf(f, "  metal activity: phase=%s (for %lld ms), completed segments=%llu, commands=%llu, encoded dispatches=%llu; "
                 "CPU wall ms: encode=%.3f, GPU/event wait=%.3f, host callback=%.3f, input prepare=%.3f, PLE gather=%.3f, shader compile=%.3f (%s)\n",
                 a.phase.load(), (long long)(a.phase_since_ms.load() ? progress_now_ms()-a.phase_since_ms.load() : 0),
                 (unsigned long long)a.segments.load(), (unsigned long long)a.commands.load(),
                 (unsigned long long)a.dispatches.load(), a.encode_us.load()/1000., a.wait_us.load()/1000.,
                 a.callback_us.load()/1000., a.prepare_us.load()/1000., a.ple_us.load()/1000., a.compile_us.load()/1000.,
                 metal_activity_timing() ? "timing enabled; cumulative" : "timing disabled");
}
inline bool metal_activity_timing() {
    static const bool on = [] { const char* v = std::getenv("STRATA_METAL_ACTIVITY"); return v && v[0] == '1'; }();
    return on;
}
// Scopes update the visible phase, never a heartbeat. Waiting is not evidence of progress.
class MetalActivityScope {
    const char* previous_;
    int64_t previous_since_;
    std::atomic<uint64_t>* elapsed_;
    std::chrono::steady_clock::time_point start_;
public:
    MetalActivityScope(const char* phase, std::atomic<uint64_t>& elapsed)
        : previous_(metal_activity().phase.exchange(phase)),
          previous_since_(metal_activity().phase_since_ms.exchange(progress_now_ms())), elapsed_(metal_activity_timing() ? &elapsed : nullptr) {
        diag_metal_fn().store(metal_activity_report);
        if (elapsed_) start_ = std::chrono::steady_clock::now();
    }
    void finish() {
        if (!previous_) return;
        if (elapsed_) elapsed_->fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now()-start_).count(), std::memory_order_relaxed);
        metal_activity().phase.store(previous_);
        metal_activity().phase_since_ms.store(previous_since_);
        previous_ = nullptr;
    }
    ~MetalActivityScope() { finish(); }
    MetalActivityScope(const MetalActivityScope&) = delete;
    MetalActivityScope& operator=(const MetalActivityScope&) = delete;
};
inline void metal_encoded(uint64_t count) {
    metal_activity().dispatches.fetch_add(count, std::memory_order_relaxed);
    progress_beat();
}
inline void metal_segment_completed() {
    metal_activity().segments.fetch_add(1, std::memory_order_relaxed);
    progress_tick();
    progress_beat();
}
inline void metal_command_completed() {
    metal_activity().commands.fetch_add(1, std::memory_order_relaxed);
    progress_beat();
}
} // namespace strata::core

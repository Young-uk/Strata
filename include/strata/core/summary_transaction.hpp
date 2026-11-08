#pragma once
#include <cstdint>
#include "strata/core/conversation_memory.hpp"
#include <charconv>
#include <optional>
#include <string>
#include <utility>

namespace strata::core {
// Protocol gate for the serialized summary transaction (version 3).
// SUMMARY_CAPS -> SUMMARY_CAPS <0|3> 16384 (configuration capability, not qualification).
// EMPTY is permitted only with explicit pristine provenance and no retained main/cache state.
// EMPTY returns zero tokens/bytes and creates no recoverable KV file.
// SUMMARY_BEGIN <max_live_tokens> <max_snapshot_bytes> <max_protected_bytes> <path> -> SUMMARY_HELD <tokens> <bytes> <ms>, after non-reclaiming SAVE.
// GEN then borrows the existing device session; checkpointing and parking are disabled.
// SUMMARY_END <same path> -> SUMMARY_RELEASED <tokens> <bytes> <ms>, only after the
// live device image and the full protected cache objects have been restored.
// SESSION/SWAIT progress and SERR/FATAL errors are the session-file protocol's.
// The caller commits DISK BEGIN only after SAVE; EMPTY bypasses SAVE.
// END commits only after full restore or checked logical-empty reset.
// Errors never implicitly release a held transaction. A failed END blocks inference.
// QUIT/EOF/crash terminates the process: the disk image survives, but protected caches
// are process-local. No promise of crash-durable preservation of the full cache chain.
// This is not a concurrent physical KV slot. The frontend must serialize the lane,
// STOP/drain summary generation before END, and retain the snapshot on any uncertainty.
// Qualification limits are per model/backend/configuration, not inferred from one short test.
// Retained memory is already resident in available-RAM telemetry: bound it separately,
// never subtract it twice. The 1 GiB scratch margin is a safety budget, not a reservation.
struct SummaryLimits {
    uint64_t tokens = 0, snapshot = 0, protected_bytes = 0;
    static constexpr uint64_t scratch = uint64_t(1) << 30;
    static uint64_t add(uint64_t a, uint64_t b) noexcept { return a > UINT64_MAX - b ? UINT64_MAX : a + b; }
    bool parse(std::string& args) {
        size_t offset = 0;
        for (uint64_t* value : {&tokens, &snapshot, &protected_bytes}) {
            const auto end = args.find(' ', offset);
            if (end == std::string::npos || end == offset) return false;
            auto result = std::from_chars(args.data() + offset, args.data() + end, *value);
            if (result.ec != std::errc{} || result.ptr != args.data() + end || *value == 0 || *value > INT64_MAX)
                return false;
            offset = end + 1;
        }
        args.erase(0, offset);
        return !args.empty();
    }
    bool covers(uint64_t live, uint64_t file, uint64_t retained) const noexcept {
        return tokens && snapshot && protected_bytes && live <= tokens && file <= snapshot && retained <= protected_bytes;
    }
    // Candidate bounded text-cache class; qualification still requires measured
    // token/snapshot/protected byte envelopes and a paired live-model test.
    static bool shape_admits(uint64_t images, bool embeddings, bool nonidentity_positions,
                             uint64_t checkpoints, uint64_t parked, uint64_t configured_slots) noexcept {
        return !images && !embeddings && !nonidentity_positions && checkpoints <= 6 &&
               parked <= 4 && parked <= configured_slots;
    }
    static bool health_admits(const std::optional<SummaryHostHealth>& before,
                              const std::optional<SummaryHostHealth>& now) noexcept {
        return before && now && before->pressure == 1 && now->pressure == 1 &&
               now->swap_used <= before->swap_used && now->swapouts == before->swapouts;
    }
    static uint64_t rollback_need(uint64_t read_peak) noexcept { return add(read_peak, scratch); }
    static bool admits(std::optional<uint64_t> available, uint64_t read_peak, uint64_t floor) noexcept {
        const auto need = rollback_need(read_peak);
        return available && need != UINT64_MAX && floor <= *available && need <= *available - floor;
    }
};
class SummaryTransaction {
    std::string path_;
    bool restoring_ = false;
    bool empty_ = false;
public:
    static constexpr int64_t context = 16384;
    bool empty_mode() const noexcept { return active() && empty_; }
    static bool invalidates_pristine(const std::string& line) noexcept {
        return line.rfind("GEN ", 0) == 0 || line.rfind("GENI ", 0) == 0 ||
               line.rfind("BGEN ", 0) == 0 || line.rfind("BGENI ", 0) == 0 || line.rfind("RESTORE ", 0) == 0;
    }
    static bool empty_eligible(bool pristine, bool live_ok, uint64_t live, uint64_t checkpoints,
                               uint64_t parked, uint64_t retained, uint64_t images, bool poisoned) noexcept {
        return pristine && !live_ok && !live && !checkpoints && !parked && !retained && !images && !poisoned;
    }
    bool active() const noexcept { return !path_.empty(); }
    bool can_begin(const std::string& path) const noexcept { return !active() && !path.empty(); }
    bool can_end(const std::string& path) const noexcept { return active() && path == path_; }
    bool allows(const std::string& line) const noexcept {
        return !active() || line == "SUMMARY_CAPS" || line == "QUIT" ||
               line.rfind("SUMMARY_END ", 0) == 0 || (!restoring_ && line.rfind("GEN ", 0) == 0);
    }
    static bool bounded(int64_t prompt, int64_t output) noexcept {
        return prompt >= 2 && output > 0 && output <= context - 8 && prompt <= context - 8 - output;
    }
    void commit_begin(std::string&& path, bool empty = false) noexcept { path_.swap(path); restoring_ = false; empty_ = empty; }
    void begin_restore() noexcept { restoring_ = true; }
    void commit_end() noexcept { path_.clear(); restoring_ = false; empty_ = false; }
};
} // namespace strata::core

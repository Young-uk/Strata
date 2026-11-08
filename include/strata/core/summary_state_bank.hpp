#pragma once
#include <charconv>
#include <cstdint>
#include <string>

namespace strata::core {
// Experimental protocol 4. Separate command names forbid accidental protocol-3 disk fallback.
struct SummaryStateBank {
    std::string id;
    uint64_t epoch = 0, main_tokens = 0, allocated_bytes = 0;
    static bool warmup_ready(bool streaming, bool arena_ready, bool gemm_enabled,
                             bool prompt_windows, bool gemm_ready, int prompt_capacity, int chunk) noexcept {
        return streaming ? arena_ready && (prompt_windows || !gemm_enabled || gemm_ready)
                         : prompt_windows || prompt_capacity >= chunk;
    }
    bool active() const noexcept { return !id.empty(); }
    static bool valid_id(const std::string& s) noexcept {
        if (s.empty() || s.size() > 64) return false;
        for (unsigned char c : s)
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'))) return false;
        return true;
    }
    static bool parse_begin(const std::string& args, uint64_t& budget, std::string& txid) {
        const auto at = args.find(' ');
        if (at == std::string::npos || at == 0 || args[0] == '0') return false;
        uint64_t n = 0;
        const auto p = std::from_chars(args.data(), args.data() + at, n);
        if (p.ec != std::errc{} || p.ptr != args.data() + at || !n || n > 9007199254740991ull) return false;
        std::string candidate = args.substr(at + 1);
        if (!valid_id(candidate)) return false;
        budget = n; txid = std::move(candidate); return true;
    }
    bool allows(const std::string& line) const noexcept {
        return !active() || line.rfind("GEN ", 0) == 0 || line.rfind("SUMMARY_STATE_END ", 0) == 0 ||
               line == "SUMMARY_STATE_CAPS" || line == "ASYNC_CAPS" ||
               line.rfind("AGEN ", 0) == 0 || line.rfind("ANEXT ", 0) == 0 || line.rfind("ACANCEL ", 0) == 0 ||
               line == "STOP" || line == "QUIT";
    }
};
}

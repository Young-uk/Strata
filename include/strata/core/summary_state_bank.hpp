#pragma once
#include <charconv>
#include <cstdint>
#include <string>
#include <vector>
#include <sstream>

namespace strata::core {
// Experimental protocol 4. Separate command names forbid accidental protocol-3 disk fallback.
struct SummaryStateBank {
    std::string id;
    uint64_t epoch = 0, main_tokens = 0, allocated_bytes = 0, context = 16384, clone_tokens = 0;
    static bool warmup_ready(bool streaming, bool arena_ready, bool gemm_enabled,
                             bool prompt_windows, bool gemm_ready, int prompt_capacity, int chunk) noexcept {
        return streaming ? arena_ready && (prompt_windows || !gemm_enabled || gemm_ready)
                         : prompt_windows || prompt_capacity >= chunk;
    }
    // Prepare shared resources without a dummy foreground inference. Publication
    // remains the caller's job, after this gate and all bank allocations succeed.
    template<class Ready, class Admit, class Prepare>
    static bool prepare_resources(Ready ready, Admit admit, Prepare prepare, std::string& err) {
        if (ready()) return true;
        if (!admit()) { err = "pressure or memory refused shared resource preparation"; return false; }
        if (!prepare()) return false;
        if (!admit()) { err = "pressure or memory changed during shared resource preparation"; return false; }
        if (!ready()) { err = "shared summary resources are not ready after preparation"; return false; }
        return true;
    }
    // Binding a fork to committed token IDs prevents a queued job from cloning
    // a different conversation after foreground work has advanced or switched.
    static uint64_t token_hash(const std::vector<int32_t>& ids) noexcept {
        uint64_t h = 14695981039346656037ull;
        for (int32_t token : ids) for (unsigned shift = 0; shift < 32; shift += 8) {
            h ^= (uint32_t(token) >> shift) & 255u; h *= 1099511628211ull;
        }
        return h;
    }
    static bool parse_fork(const std::string& args, uint64_t& budget, std::string& id,
                           uint64_t& tokens, uint64_t& hash) {
        std::istringstream in(args); std::string b, candidate, n, h, extra;
        if (!(in >> b >> candidate >> n >> h) || (in >> extra) ||
            !parse_begin(b + " " + candidate, budget, id)) return false;
        auto number = [](const std::string& s, uint64_t& out) {
            if (s.empty() || (s.size() > 1 && s[0] == '0')) return false;
            const auto p = std::from_chars(s.data(), s.data()+s.size(), out);
            return p.ec == std::errc{} && p.ptr == s.data()+s.size();
        };
        return number(n, tokens) && tokens > 0 && number(h, hash);
    }
    static bool parse_checkpoint(const std::string& args, uint64_t& budget, std::string& id,
                                 uint64_t& tokens, uint64_t& hash, uint64_t& prefix, uint64_t& prefix_hash) {
        std::istringstream in(args);
        std::string b, name, n, h, p, ph, extra;
        if (!(in >> b >> name >> n >> h >> p >> ph) || (in >> extra) ||
            !parse_fork(b + " " + name + " " + n + " " + h, budget, id, tokens, hash)) return false;
        uint64_t ignored = 0; std::string ignored_id;
        return parse_fork("1 Prefix " + p + " " + ph, ignored, ignored_id, prefix, prefix_hash)
            && prefix < tokens;
    }
    static bool physical_capacity(uint64_t logical, uint64_t reserve, uint64_t& cells) noexcept {
        if (reserve > 8192 || logical > uint64_t(INT32_MAX) - reserve - 128) return false;
        const uint64_t rounded = (logical + reserve + 64) / 4 * 4 + 64;
        cells = rounded < 4096 ? 4096 : rounded;
        return true;
    }
    static bool clone_room(uint64_t next, uint64_t input, uint64_t output, uint64_t context,
                           uint64_t physical_context = 0) noexcept {
        // Reused absolute positions have bounded headroom even after compaction.
        const uint64_t capacity = physical_context ? physical_context : context;
        return context > 0 && context <= INT32_MAX && capacity >= context &&
               capacity <= INT32_MAX && capacity <= 4 * context && input <= capacity && output <= capacity &&
               next < 4 * context && input + output + 8 < 4 * context - next;
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
               line == "SUMMARY_STATE_CAPS" || line == "SUMMARY_CLONE_CAPS" || line == "ASYNC_CAPS" ||
               line.rfind("AFORK ", 0) == 0 ||
               line.rfind("AGEN ", 0) == 0 || line.rfind("ANEXT ", 0) == 0 || line.rfind("ACANCEL ", 0) == 0 ||
               line == "STOP" || line == "QUIT";
    }
};
}

#pragma once
#include "strata/core/summary_state_bank.hpp"
#include "strata/kernels/sampler.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <vector>

namespace strata::core {
// A cursor is a count of COMMITTED inputs. ids[cursor] is the pending input,
// including the last emitted token. A yield never destroys this record.
struct NativeContinuation {
    std::string id;
    bool summary = false, done = false;
    size_t origin = 0, cursor = 0, produced = 0, limit = 0;
    std::vector<int32_t> ids;
    strata::kernels::SamplerParams sampler;
    size_t penalty_origin = 0;
    bool cache = true;
    int logprobs = -1;
    std::vector<float> bias;
    std::vector<int32_t> allowed;
    // Failed/cancelled admission must never report another conversation's live
    // length. Only exact current-request progress is a committed cursor.
    static size_t preparation_cursor(size_t prompt, int64_t reached, int64_t resume, bool exact) {
        const int64_t current = exact ? reached : resume;
        return current <= 0 ? 0 : std::min(prompt, (size_t)current);
    }
    bool active() const { return !id.empty() && !done; }
    size_t prompt_remaining() const { return cursor < origin - 1 ? origin - 1 - cursor : 0; }
    size_t quantum_inputs(size_t summary_inputs = 1) const {
        const size_t bound = summary ? std::clamp<size_t>(summary_inputs, 1, 128) : 512;
        return prompt_remaining() ? std::min<size_t>(bound, prompt_remaining()) : 1;
    }
    // Protocol 2: an optional bounded input budget for this quantum only.
    // Legacy ANEXT <id> keeps its original one-input summary behavior.
    static bool parse_step(const std::string& args, std::string& id, size_t& inputs) {
        std::istringstream in(args);
        std::string value, extra;
        inputs = 1;
        if (!(in >> id) || !SummaryStateBank::valid_id(id)) return false;
        if (!(in >> value)) return true;
        uint64_t n = 0;
        auto parsed = std::from_chars(value.data(), value.data() + value.size(), n);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || !n || n > 128 || (in >> extra)) return false;
        inputs = (size_t)n;
        return true;
    }
    static bool parse(const std::string& args, int64_t vocab, NativeContinuation& result, std::string& error) {
        NativeContinuation c;
        std::istringstream in(args); std::string lane, count, word;
        auto fail = [&]() { error = "invalid or unsupported continuation arguments"; return false; };
        if (!(in >> c.id >> lane >> count) || !SummaryStateBank::valid_id(c.id) || (lane != "main" && lane != "summary")) return fail();
        c.summary = lane == "summary";
        uint64_t n = 0;
        auto integer = [](const std::string& s, uint64_t& v) {
            if (s.empty()) return false;
            auto p = std::from_chars(s.data(), s.data() + s.size(), v);
            return p.ec == std::errc{} && p.ptr == s.data() + s.size();
        };
        if (!integer(count, n) || n == 0 || n > INT32_MAX) return fail();
        c.limit = (size_t)n;
        c.sampler.greedy = true; c.sampler.temperature = 0; c.sampler.top_p = 1;
        c.sampler.top_k = 20; c.sampler.penalty_last_n = 0; c.sampler.penalty_repeat = 1;
        c.sampler.penalty_freq = c.sampler.penalty_present = 0; c.sampler.min_p = 0;
        c.sampler.seed = 0; c.sampler.counter = 0;
        bool tokens = false;
        while (in >> word) {
            const auto eq = word.find('=');
            if (eq != std::string::npos) {
                if (tokens) return fail();
                const std::string key = word.substr(0, eq), val = word.substr(eq + 1);
                if (key == "seed") { if (!integer(val, n)) return fail(); c.sampler.seed = n; continue; }
                char* end = nullptr; const float f = std::strtof(val.c_str(), &end);
                if (val.empty() || *end || !std::isfinite(f)) return fail();
                if (key == "temperature" && f >= 0) c.sampler.temperature = f;
                else if (key == "top_p" && f > 0 && f <= 1) c.sampler.top_p = f;
                else if (key == "min_p" && f >= 0 && f <= 1) c.sampler.min_p = f;
                else if (key == "penalty_repeat" && f > 0) c.sampler.penalty_repeat = f;
                else if (key == "penalty_freq") c.sampler.penalty_freq = f;
                else if (key == "penalty_present") c.sampler.penalty_present = f;
                else if (key == "top_k" && integer(val, n) && n >= 1 && n <= 64) c.sampler.top_k = (int)n;
                else if (key == "penalty_last_n" && integer(val, n) && n <= 4096) c.sampler.penalty_last_n = (int)n;
                else return fail();
            } else {
                if (tokens) return fail();
                tokens = true; size_t a = 0;
                do {
                    const auto b = word.find(',', a);
                    if (!integer(word.substr(a, b == std::string::npos ? b : b - a), n) || n >= (uint64_t)vocab) return fail();
                    c.ids.push_back((int32_t)n);
                    if (b == std::string::npos) break;
                    a = b + 1;
                } while (true);
            }
        }
        if (!tokens || c.ids.empty()) return fail();
        c.origin = c.ids.size(); c.penalty_origin = c.origin - 1; c.sampler.greedy = c.sampler.temperature <= 0;
        result = std::move(c); return true;
    }
};
}

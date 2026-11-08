#include "strata/core/native_continuation.hpp"
#include <cassert>
#include <iostream>
int main() {
    using strata::core::NativeContinuation;
    NativeContinuation c; std::string e;
    assert(NativeContinuation::preparation_cursor(20, 8, 3, true) == 8);
    assert(NativeContinuation::preparation_cursor(20, 8, 3, false) == 3);
    assert(NativeContinuation::preparation_cursor(20, 1000, 1000, true) == 20);
    assert(NativeContinuation::preparation_cursor(20, -1, -1, false) == 0);
    assert(NativeContinuation::preparation_cursor(20, 0, 0, true) == 0);
    // An outgoing 1000-token live conversation cannot enter this calculation.

    assert(NativeContinuation::parse("M1 main 4 temperature=0.7 seed=123 penalty_last_n=64 1,2,3", 100, c, e));
    assert(c.active() && !c.summary && c.origin == 3 && c.cursor == 0 && c.limit == 4);
    assert(c.sampler.seed == 123 && !c.sampler.greedy && c.prompt_remaining() == 2);
    assert(c.quantum_inputs() == 2);
    {
        NativeContinuation long_prompt; long_prompt.origin = 1001;
        assert(long_prompt.quantum_inputs() == 512);
        long_prompt.summary = true; assert(long_prompt.quantum_inputs() == 1);
        long_prompt.cursor = 999; assert(long_prompt.quantum_inputs() == 1);
        long_prompt.cursor = 1000; assert(long_prompt.quantum_inputs() == 1);
    }
    {
        NativeContinuation idle; idle.summary = true; idle.origin = 1001;
        assert(idle.quantum_inputs(128) == 128);
        assert(idle.quantum_inputs(1) == 1);
        assert(idle.quantum_inputs(999) == 128);
        idle.cursor = 990; assert(idle.quantum_inputs(128) == 10);
        idle.cursor = 1000; assert(idle.quantum_inputs(128) == 1); // decode never batches
        std::string id; size_t inputs = 0;
        assert(NativeContinuation::parse_step("ABC123", id, inputs) && inputs == 1);
        assert(NativeContinuation::parse_step("ABC123 128", id, inputs) && inputs == 128);
        for (const char* bad : {"", "a-b 128", "A 0", "A 129", "A -1", "A nan", "A 1 2"})
            assert(!NativeContinuation::parse_step(bad, id, inputs));
    }
    c.cursor = 2; assert(c.prompt_remaining() == 0);
    c.cursor++; c.ids.push_back(7); c.produced++;
    const auto preserved = c;
    NativeContinuation summary;
    assert(NativeContinuation::parse("S1 summary 10 5,6", 100, summary, e));
    assert(summary.quantum_inputs() == 1);
    summary.cursor++; summary.cursor++; summary.ids.push_back(8); summary.produced++;
    assert(c.ids == preserved.ids && c.cursor == preserved.cursor && c.sampler.seed == preserved.sampler.seed);
    assert(c.ids[c.cursor] == 7 && c.origin == 3); // emitted but not consumed
    for (const char* bad : {"M1 main 0 1", "M1 other 1 1", "M1 main 1 temperature=nan 1", "M1 main 1 seed=-1 1", "M1 main 1 top_k=0 1", "M1 main 1 top_k=65 1", "M1 main 1 pin=3 1", "M1 main 1 1,", "M1 main 1 100", "M1 main 1 -1", "M1 main 1 1 2", "M1 main 1 1 seed=3"})
        assert(!NativeContinuation::parse(bad, 100, summary, e));
    // Prepared first-token capture: non-default origin, bias and cache policy
    // belong to the continuation and survive an unrelated lane's admission.
    c.penalty_origin = 1; c.bias = {0.0f, -2.0f}; c.cache = false;
    c.logprobs = 3; c.allowed = {1, 2};
    auto prepared = c;
    assert(prepared.ids[prepared.cursor] == 7 && prepared.produced == 1);
    assert(prepared.penalty_origin == 1 && !prepared.cache && prepared.bias[1] == -2.0f);
    assert(prepared.logprobs == 3 && prepared.allowed == c.allowed);
    c.done = true; assert(!c.active());
    std::cout << "native_continuation_test: passed\n";
}

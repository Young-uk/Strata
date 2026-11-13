#include "strata/core/metal_activity.hpp"
#include <cassert>
#include <cstring>
using namespace strata::core;
int main() {
    const auto before = progress().beats.load();
    {
        MetalActivityScope waiting("test wait", metal_activity().wait_us);
        for (int i = 0; i < 10000; ++i) (void)metal_activity().segments.load();
        assert(progress().beats.load() == before); // polling must never mask a stalled GPU
        {
            MetalActivityScope callback("test callback", metal_activity().callback_us);
            assert(std::strcmp(metal_activity().phase.load(), "test callback") == 0);
        }
        assert(std::strcmp(metal_activity().phase.load(), "test wait") == 0);
        waiting.finish();
        waiting.finish(); // no double accounting or phase restoration
    }
    assert(std::strcmp(metal_activity().phase.load(), "idle") == 0);
    assert(progress().beats.load() == before);
    metal_encoded(128);
    assert(progress().beats.load() == before + 1);
    const auto ticks = progress().ticks.load();
    metal_segment_completed();
    assert(progress().ticks.load() == ticks + 1);
    assert(progress().beats.load() == before + 2);
    metal_command_completed();
    assert(progress().beats.load() == before + 3);
    assert(metal_activity().dispatches.load() == 128);
    assert(metal_activity().segments.load() == 1);
    assert(metal_activity().commands.load() == 1);
    assert(diag_metal_fn().load() == metal_activity_report);
    metal_activity_report(stdout);
}

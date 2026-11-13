// Exercise the actual command/event handshake without loading a model.
#include "mtl_rt.h"
#include "strata/core/metal_activity.hpp"
#include <cassert>
#include <fstream>
#include <filesystem>
#include <unistd.h>
using namespace strata::core;
struct CallbackState { mtl::Buffer* buffer; int calls = 0; };
int main() {
    auto path = std::filesystem::temp_directory_path() / ("strata-activity-" + std::to_string(getpid()) + ".metal");
    { std::ofstream f(path); f << "kernel void inc(device uint* x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i] += 1; }\n"
          << "kernel void inc_alt(device uint* x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i] += 1; }"; }
    mtl::Context context;
    context.ggml_common_path.clear();
    int lib = context.compile({path.string()});
    std::filesystem::remove(path);
    auto pipeline = context.pipeline(lib, "inc");
    auto alternate = context.pipeline(lib, "inc_alt");
    auto buffer = context.alloc(32 * sizeof(uint32_t));
    mtl::Dispatch d;
    d.p = &pipeline;
    d.args = {mtl::Arg::B(buffer)};
    std::vector<mtl::Dispatch> graph(259, d);
    for (size_t i = 1; i < graph.size(); i += 2) graph[i].p = &alternate; // exact dispatch count, including the final partial batch
    auto check = [&](uint32_t value) {
        for (int i = 0; i < 32; ++i) assert(buffer.as<uint32_t>()[i] == value);
    };
    context.run_graph(graph);
    check(259);
    auto token = context.run_graph_async(graph);
    context.wait_async(token);
    check(518);
    CallbackState state{&buffer};
    context.run_graph_seg(graph, {1, 130, 258}, [](void* p, int s) {
        auto& state = *static_cast<CallbackState*>(p);
        assert(s == state.calls++);
        const uint32_t expected[] = {519, 648, 776};
        for (int i = 0; i < 32; ++i) assert(state.buffer->as<uint32_t>()[i] == expected[s]);
        assert(metal_activity().segments.load() == static_cast<uint64_t>(s + 1));
    }, &state);
    check(777);
    assert(state.calls == 3);
    assert(metal_activity().dispatches.load() == 259 * 3);
    assert(metal_activity().commands.load() == 3);
    assert(metal_activity().segments.load() == 3);
    assert(std::string(metal_activity().phase.load()) == "idle");
    metal_activity_report(stdout);
    context.free(buffer);
    std::puts("Metal graph, async and segmented handshake: PASS (all values exact)");
}

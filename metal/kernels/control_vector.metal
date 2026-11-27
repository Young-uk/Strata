#include <metal_stdlib>
using namespace metal;

struct ControlVectorArgs {
    uint width, streams, mode;
    float scale;
};

// Apply the pending FFN update, then steer the completed residual of this layer.
kernel void control_vector(device float* residual [[buffer(0)]],
                           device const float* block [[buffer(1)]],
                           device const float* gates [[buffer(2)]],
                           device const float* direction [[buffer(3)]],
                           constant ControlVectorArgs& args [[buffer(4)]],
                           uint3 group [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]],
                           ushort simd [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[8];
    device float* row = residual + (group.y * args.streams + group.x) * args.width;
    device const float* update = block + group.y * args.width;
    const float gate = 2.0f / (1.0f + exp(-gates[group.y * args.streams + group.x] / float(args.streams)));
    float dot = 0.0f;
    for (uint d = tid; d < args.width; d += 256) {
        const float value = fma(update[d], gate, row[d]);
        row[d] = value;
        if (args.mode == 0) dot = fma(value, direction[d], dot);
    }
    if (args.mode == 0) {
        dot = simd_sum(dot);
        if (lane == 0) partial[simd] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        dot = 0.0f;
        for (uint i = 0; i < 8; ++i) dot += partial[i];
        dot *= args.scale;
        for (uint d = tid; d < args.width; d += 256) row[d] = fma(-dot, direction[d], row[d]);
    } else {
        for (uint d = tid; d < args.width; d += 256) row[d] += direction[d];
    }
}

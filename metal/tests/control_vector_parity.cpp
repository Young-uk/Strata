#include "mtl_rt.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

int main() {
    constexpr uint32_t width = 2560, streams = 4, tokens = 2;
    constexpr size_t rows = streams * tokens;
    mtl::Context metal;
    const auto library = metal.compile({"control_vector.metal"}, "#define NO_GGML_COMMON 1\n");
    const auto pipeline = metal.pipeline(library, "control_vector");
    auto residual = metal.alloc(rows * width * sizeof(float));
    auto block = metal.alloc(tokens * width * sizeof(float));
    auto gates = metal.alloc(rows * sizeof(float));
    auto direction = metal.alloc(width * sizeof(float));
    std::vector<float> original(rows * width);
    for (size_t i = 0; i < original.size(); ++i) original[i] = float(i % 17) / 17 - .5f;
    for (size_t i = 0; i < tokens * width; ++i) block.as<float>()[i] = float(i % 11) / 11 - .5f;
    for (size_t i = 0; i < rows; ++i) gates.as<float>()[i] = float(i) - 3;
    for (size_t i = 0; i < width; ++i) direction.as<float>()[i] = (i % 2 ? -1.f : 1.f) / std::sqrt(float(width));
    for (uint32_t mode : {0u, 1u}) {
        struct Args { uint32_t width, streams, mode; float scale; };
        const Args args{width, streams, mode, 1.0f};
        std::copy(original.begin(), original.end(), residual.as<float>());
        mtl::Dispatch dispatch;
        dispatch.p = &pipeline;
        dispatch.args = {mtl::Arg::B(residual), mtl::Arg::B(block), mtl::Arg::B(gates), mtl::Arg::B(direction), mtl::Arg::V(args)};
        dispatch.grid[0] = streams;
        dispatch.grid[1] = tokens;
        dispatch.tg[0] = 256;
        metal.run_graph({dispatch});
        float error = 0;
        for (size_t row = 0; row < rows; ++row) {
            const float g = 2.f / (1.f + std::exp(-gates.as<float>()[row] / streams));
            auto input = [&](size_t d) {return std::fma(block.as<float>()[row / streams * width + d], g, original[row * width + d]);};
            double dot = 0;
            for (size_t d = 0; d < width; ++d) dot += double(input(d)) * direction.as<float>()[d];
            for (size_t d = 0; d < width; ++d) {
                const float expected = mode == 0 ? std::fma(-float(dot), direction.as<float>()[d], input(d)) : input(d) + direction.as<float>()[d];
                const float actual = residual.as<float>()[row * width + d];
                if (!std::isfinite(actual)) return 1;
                error = std::max(error, std::abs(expected - actual));
            }
        }
        std::printf("%s %s max absolute error: %.8g\n", metal.device_name().c_str(), mode == 0 ? "projection" : "addition", error);
        if (error > 2e-6f) return 1;
    }
}

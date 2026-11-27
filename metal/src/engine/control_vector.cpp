#include "strata/kernels/cvec.hpp"

namespace strata::kernels {
namespace {
Cvec vector;
std::vector<float> directions;
std::vector<float> scales;
bool enabled = false;
}

const Cvec& cvec() { return vector; }

bool cvec_upload(const std::vector<float>& dir, const std::vector<float>& scale, int mode, int first, int last,
                 int64_t width, int64_t streams, std::string& error) {
    if (width != 2560 || streams != 4 || (mode != 0 && mode != 1) || first < 1 ||
        first > last || static_cast<size_t>(last) >= scale.size() ||
        dir.size() != scale.size() * static_cast<size_t>(width)) {
        error = "control vector: unsupported Metal Qwen geometry or invalid tables";
        return false;
    }
    directions = dir;
    scales = scale;
    // Metal windows upload these immutable host tables into their own MTLBuffer.
    vector = {directions.data(), scales.data(), nullptr, mode, first, last, width, streams, {}};
    for (float value : scales) vector.steered.push_back(value != 0.0f);
    enabled = true;
    return true;
}

bool cvec_replicate(std::string&) { return true; }
void cvec_set_enabled(bool on) { enabled = on; }
bool cvec_enabled() { return vector.loaded() && enabled; }

}  // namespace strata::kernels

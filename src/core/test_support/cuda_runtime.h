// Test-only extension of the Metal-compatible declarations for generic CPU mocks.
#pragma once
#include_next <cuda_runtime.h>
struct cudaPointerAttributes { int type = 0; };
inline constexpr int cudaMemoryTypeDevice = 2;
inline cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* at, const void*) {
    at->type = 0; // all fixture memory is host owned
    return cudaSuccess;
}

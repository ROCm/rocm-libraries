// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include <rocwmma/rocwmma.hpp>

// Compile a device-side consumer without requiring a GPU at test execution time.
__global__ void fill_tile(float* output)
{
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> tile;
    rocwmma::fill_fragment(tile, 1.0f);
    rocwmma::store_matrix_sync(output, tile, 16, rocwmma::mem_row_major);
}

int main(int argc, char**)
{
    // Volatile operands keep the host FP16 conversions from being folded away.
    volatile float              input = static_cast<float>(argc);
    volatile rocwmma::float16_t half  = static_cast<rocwmma::float16_t>(input);
    return static_cast<float>(half) == input ? 0 : 1;
}

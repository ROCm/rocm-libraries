// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Public interface for the MFMA-tiled GEMM kernel.

#pragma once

#include <hip/hip_runtime.h>

namespace rocm_ck::test {

enum class AccumKLoopPolicy
{
    Naive,
    TwoSum,
    Kahan,
};

void launch_mfma_tile_gemm(const float* A,
                           const float* B,
                           float* partial,
                           int M,
                           int N,
                           int K_full,
                           int k_offset,
                           int K_tile,
                           hipStream_t stream,
                           AccumKLoopPolicy accum_policy = AccumKLoopPolicy::Naive);

bool can_use_mfma(int M, int N, int K_tile);

} // namespace rocm_ck::test

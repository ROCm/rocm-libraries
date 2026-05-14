// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// Per-head pointers are device float arrays of length nhead_q.
struct sparge_hyperparam_args
{
    float cdfthreshd   = 1.0f;  // 1.0 = dense
    float topk         = 0.0f;
    float simthreshold = 0.0f;
    float pvthreshd    = 0.0f;  // log2 units; 0 = Stage 2 off

    const void* cdfthreshd_per_head_ptr   = nullptr;
    const void* topk_per_head_ptr         = nullptr;
    const void* simthreshold_per_head_ptr = nullptr;
    const void* pvthreshd_per_head_ptr    = nullptr;

    bool smooth_k = true;
};

__device__ __host__ inline float
lookup_per_head(const void* per_head_ptr, index_t h_q, float scalar_fallback)
{
    return per_head_ptr ? reinterpret_cast<const float*>(per_head_ptr)[h_q]
                        : scalar_fallback;
}

} // namespace ck_tile

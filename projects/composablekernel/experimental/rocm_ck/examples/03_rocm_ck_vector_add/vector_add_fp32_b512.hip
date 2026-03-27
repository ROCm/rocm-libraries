// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/elementwise_dev.hpp>

// Tile scaling: same type and warp count as fp32_b256, 2x tile.
// Doubling block_tile doubles kRepeatM (1 -> 2); threads stay at 64.
static constexpr rocm_ck::ElementwiseSpec spec = rocm_ck::make_kernel(
    rocm_ck::Signature{
        .dtype = rocm_ck::DataType::FP32,
        .ops = {rocm_ck::AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    rocm_ck::ElementwiseAlgorithm{.block_tile = 512, .block_warps = 1,
                                   .warp_tile = 512, .pad = true});

extern "C" __global__ void vector_add_fp32_b512(rocm_ck::Args args)
{
    rocm_ck::run<spec>(args);
}

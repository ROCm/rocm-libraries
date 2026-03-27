// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/elementwise_dev.hpp>

// Baseline single-warp FP32 variant. Smallest tile in the set.
// Geometry: kVectorM=4 (128-bit / 32-bit), kRepeatM=1, 64 threads.
static constexpr rocm_ck::ElementwiseSpec spec = rocm_ck::make_kernel(
    rocm_ck::Signature{
        .dtype = rocm_ck::DataType::FP32,
        .ops = {rocm_ck::AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    rocm_ck::ElementwiseAlgorithm{.block_tile = 256, .block_warps = 1,
                                   .warp_tile = 256, .pad = true});

extern "C" __global__ void vector_add_fp32_b256(rocm_ck::Args args)
{
    rocm_ck::run<spec>(args);
}

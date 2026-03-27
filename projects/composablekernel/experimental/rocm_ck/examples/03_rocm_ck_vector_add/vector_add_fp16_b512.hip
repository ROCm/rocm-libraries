// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/elementwise_dev.hpp>

// Narrower type doubles vectorization: FP16 gets kVectorM=8 (vs FP32's 4)
// because 128-bit registers hold 8 half-precision values instead of 4 floats.
static constexpr rocm_ck::ElementwiseSpec spec = rocm_ck::make_kernel(
    rocm_ck::Signature{
        .dtype = rocm_ck::DataType::FP16,
        .ops = {rocm_ck::AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    rocm_ck::ElementwiseAlgorithm{.block_tile = 512, .block_warps = 1,
                                   .warp_tile = 512, .pad = true});

extern "C" __global__ void vector_add_fp16_b512(rocm_ck::Args args)
{
    rocm_ck::run<spec>(args);
}

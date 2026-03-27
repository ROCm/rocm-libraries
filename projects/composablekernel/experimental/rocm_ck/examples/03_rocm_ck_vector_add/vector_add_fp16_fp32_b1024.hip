// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/elementwise_dev.hpp>

// Widening: FP16 inputs -> FP32 output. Tensor override sets output type.
// The wider type (FP32) constrains kVectorM to 4 (vs 8 for homogeneous FP16).
static constexpr rocm_ck::ElementwiseSpec spec = rocm_ck::make_kernel(
    rocm_ck::Signature{
        .dtype = rocm_ck::DataType::FP16,
        .tensors = {rocm_ck::Tensor{.name = "C", .dtype = rocm_ck::DataType::FP32}},
        .ops = {rocm_ck::AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    rocm_ck::ElementwiseAlgorithm{.block_tile = 1024, .block_warps = 1,
                                   .warp_tile = 1024, .pad = true});

extern "C" __global__ void vector_add_fp16_fp32_b1024(rocm_ck::Args args)
{
    rocm_ck::run<spec>(args);
}

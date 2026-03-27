// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/elementwise_dev.hpp>

// Narrowing: FP32 inputs -> FP16 output. Same kVectorM=4 constraint as widening
// (wider type always governs). Output stored in FP16 for reduced memory footprint.
static constexpr rocm_ck::ElementwiseSpec spec = rocm_ck::make_kernel(
    rocm_ck::Signature{
        .dtype = rocm_ck::DataType::FP32,
        .tensors = {rocm_ck::Tensor{.name = "C", .dtype = rocm_ck::DataType::FP16}},
        .ops = {rocm_ck::AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    rocm_ck::ElementwiseAlgorithm{.block_tile = 1024, .block_warps = 1,
                                   .warp_tile = 1024, .pad = true});

extern "C" __global__ void vector_add_fp32_fp16_b1024(rocm_ck::Args args)
{
    rocm_ck::run<spec>(args);
}

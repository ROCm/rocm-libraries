// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/elementwise_dev.hpp>

// Multi-warp FP32: 8 warps x 64 = 512 threads, 2048 elements per block.
// Small warp_tile (64) forces kVectorM=1 (minimum vectorization) — contrast
// with fp16_b1024_w2 where the narrow type allows kVectorM=8.
static constexpr rocm_ck::ElementwiseSpec spec = rocm_ck::make_kernel(
    rocm_ck::Signature{
        .dtype = rocm_ck::DataType::FP32,
        .ops = {rocm_ck::AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    rocm_ck::ElementwiseAlgorithm{.block_tile = 2048, .block_warps = 8,
                                   .warp_tile = 64, .pad = true});

extern "C" __global__ void vector_add_fp32_b2048_w8(rocm_ck::Args args)
{
    rocm_ck::run<spec>(args);
}

// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocm_ck/elementwise_dev.hpp>

// Multi-warp FP16: 2 warps x 64 = 128 threads, same block_tile as fp16_b1024.
// Larger warp_tile (512) preserves kVectorM=8 (max for 16-bit). Compare with
// single-warp fp16_b1024: same work, but kRepeatM drops from 2 to 1.
static constexpr rocm_ck::ElementwiseSpec spec = rocm_ck::make_kernel(
    rocm_ck::Signature{
        .dtype = rocm_ck::DataType::FP16,
        .ops = {rocm_ck::AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    rocm_ck::ElementwiseAlgorithm{.block_tile = 1024, .block_warps = 2,
                                   .warp_tile = 512, .pad = true});

extern "C" __global__ void vector_add_fp16_b1024_w2(rocm_ck::Args args)
{
    rocm_ck::run<spec>(args);
}

// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// fp16 GEMM kernel variant with 32×32 warp tile. Demonstrates a different
// tile configuration: wider warp tiles process more elements per MFMA
// instruction at the cost of fewer warp-level iterations per block tile.

#include <rocm_ck/gemm_dev.hpp>

static constexpr rocm_ck::GemmSpec spec = rocm_ck::make_kernel(
    rocm_ck::Signature{
        .dtype = rocm_ck::DataType::FP16,
        .ops = {rocm_ck::GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    rocm_ck::GemmAlgorithm{.block_tile  = {128, 128, 32},
                            .block_warps = {2, 2, 1},
                            .warp_tile   = {32, 32, 16}});

extern "C" __global__ void gemm_fp16_w32(rocm_ck::Args args)
{
    rocm_ck::run<spec>(args);
}

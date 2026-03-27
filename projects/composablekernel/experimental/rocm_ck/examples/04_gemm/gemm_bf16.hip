// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// bf16 GEMM kernel variant with 16×16×16 warp tile. Accumulates in fp32,
// stores bf16 output via CShuffleEpilogue.

#include <rocm_ck/gemm_dev.hpp>

static constexpr rocm_ck::GemmSpec spec = rocm_ck::make_kernel(
    rocm_ck::Signature{
        .dtype = rocm_ck::DataType::BF16,
        .ops = {rocm_ck::GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    rocm_ck::GemmAlgorithm{.block_tile  = {128, 128, 32},
                            .block_warps = {2, 2, 1},
                            .warp_tile   = {16, 16, 16}});

extern "C" __global__ void gemm_bf16(rocm_ck::Args args)
{
    rocm_ck::run<spec>(args);
}

// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// fp16 GEMM + D0 addition + ReLU: E = relu(A*B + D0). Demonstrates composed
// epilogue with bias addition and fused activation. Accumulates in fp32,
// stores fp16 output via CShuffleEpilogue.

#include <rocm_ck/gemm_dev.hpp>

static constexpr rocm_ck::GemmSpec spec = rocm_ck::make_kernel(
    rocm_ck::Signature{
        .dtype = rocm_ck::DataType::FP16,
        .ops = {rocm_ck::GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                rocm_ck::AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
                rocm_ck::ReluOp{.in = "D", .out = "E"}}},
    rocm_ck::GemmAlgorithm{.block_tile  = {128, 128, 32},
                            .block_warps = {2, 2, 1},
                            .warp_tile   = {16, 16, 16}});

extern "C" __global__ void gemm_fp16_add_relu(rocm_ck::Args args)
{
    rocm_ck::run<spec>(args);
}

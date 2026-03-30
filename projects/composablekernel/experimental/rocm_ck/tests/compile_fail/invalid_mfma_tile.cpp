// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: FP32 with 32x32x16 MFMA tile (k=16 invalid for FP32 at 32x32).
// Expected error: "mfma_tile is not a valid MFMA instruction shape for this dtype"

#include <rocm_ck/gemm_spec.hpp>

using namespace rocm_ck;

constexpr auto bad = make_spec(
    Signature{.dtype = DataType::FP32, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {32, 32, 16}});

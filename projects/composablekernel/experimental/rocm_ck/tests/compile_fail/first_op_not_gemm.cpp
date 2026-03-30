// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: first op is not GemmOp.
// Expected error: "GEMM make_spec requires GemmOp as first operator"

#include <rocm_ck/gemm_spec.hpp>

using namespace rocm_ck;

constexpr auto bad = make_spec(
    Signature{.dtype = DataType::FP16, .ops = {AddOp{.lhs = "A", .rhs = "B", .out = "C"}}},
    GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}});

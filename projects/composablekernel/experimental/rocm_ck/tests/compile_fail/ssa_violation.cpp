// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: two ops output the same tensor name (SSA violation).
// Expected error: "SSA violation: tensor produced by multiple operators"

#include <rocm_ck/gemm_spec.hpp>

using namespace rocm_ck;

constexpr auto bad = resolve(Signature{.dtype = DataType::FP16,
                                       .ops   = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
                                                 AddOp{.lhs = "X", .rhs = "Y", .out = "C"}}});

// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: empty tensor name in operator slot.
// Expected error: "operator slot has empty tensor name"

#include <rocm_ck/gemm_spec.hpp>

using namespace rocm_ck;

constexpr auto bad =
    resolve(Signature{.dtype = DataType::FP16, .ops = {GemmOp{.lhs = "", .rhs = "B", .out = "C"}}});

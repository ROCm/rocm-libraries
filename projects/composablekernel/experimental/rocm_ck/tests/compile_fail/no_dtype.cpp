// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: no dtype set (neither on Signature nor individual tensors).
// Expected error: "tensor dtype unresolvable: set tensor dtype or signature dtype"

#include <rocm_ck/gemm_spec.hpp>

using namespace rocm_ck;

constexpr auto bad = resolve(Signature{.ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}});

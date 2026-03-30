// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Must fail: Preshuffle pipeline requires A=RowMajor.
// Expected error: "Preshuffle pipeline requires A layout = Row"

#include <rocm_ck/gemm_spec.hpp>

using namespace rocm_ck;

constexpr auto bad =
    make_spec(Signature{.dtype   = DataType::FP16,
                        .tensors = {Tensor{.name = "A", .layout = Layout::Col}},
                        .ops     = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}},
              GemmAlgorithm{{128, 128, 32}, {2, 2, 1}, {16, 16, 16}, 1, Pipeline::Preshuffle});

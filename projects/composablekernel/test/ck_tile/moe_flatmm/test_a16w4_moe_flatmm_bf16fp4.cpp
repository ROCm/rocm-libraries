// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "test_a16w4_moe_flatmm_base.hpp"
#include "test_moe_flatmm_scenarios.hpp"

// A16W4 MoE FlatMM bf16xfp4 on gfx950 (F16xMXF4FlatmmPipelineAGmemBGmemCRegV1).
// Mirrors AITER's `a16w4_gfx950` codegen path (bf16 activation, pk_fp4 weights
// with K=32 MX block scales, bf16 output, Swiglu activation, expert bias).
// Covers gate_only as a support probe in addition to the AITER gate_up/gemm2 path.
// clang-format off
using BF16FP4Types = ::testing::Types<
    std::tuple<A16W4_GemmTypeConfig_bf16xfp4, GateOnly>,
    std::tuple<A16W4_GemmTypeConfig_bf16xfp4, GateUp>,
    std::tuple<A16W4_GemmTypeConfig_bf16xfp4, Gemm2>>;
// clang-format on

template <typename Tuple>
class TestA16W4MoeFlatmmBF16FP4 : public TestA16W4MoeFlatmmBase<Tuple>
{
};

TYPED_TEST_SUITE(TestA16W4MoeFlatmmBF16FP4, BF16FP4Types);

// N must be at least 2 * N_Tile = 512 because gate_up uses outputN = N/2 and
// outputN has to remain a multiple of N_Tile (taken care of by the shared
// scenario set, which uses N=512 throughout).
MOE_FLATMM_DECLARE_SCENARIOS(TestA16W4MoeFlatmmBF16FP4)

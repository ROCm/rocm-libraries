// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "test_a16w4_moe_flatmm_base.hpp"
#include "test_moe_flatmm_scenarios.hpp"

// A16W4 MoE FlatMM fp16xfp4 on gfx950 (F16xMXF4FlatmmPipelineAGmemBGmemCRegV1).
// Same shape coverage as the bf16xfp4 suite, including gate_only as a support probe.
// clang-format off
using FP16FP4Types = ::testing::Types<
    std::tuple<A16W4_GemmTypeConfig_fp16xfp4, GateOnly>,
    std::tuple<A16W4_GemmTypeConfig_fp16xfp4, GateUp>,
    std::tuple<A16W4_GemmTypeConfig_fp16xfp4, Gemm2>>;
// clang-format on

template <typename Tuple>
class TestA16W4MoeFlatmmFP16FP4 : public TestA16W4MoeFlatmmBase<Tuple>
{
};

TYPED_TEST_SUITE(TestA16W4MoeFlatmmFP16FP4, FP16FP4Types);

MOE_FLATMM_DECLARE_SCENARIOS(TestA16W4MoeFlatmmFP16FP4)

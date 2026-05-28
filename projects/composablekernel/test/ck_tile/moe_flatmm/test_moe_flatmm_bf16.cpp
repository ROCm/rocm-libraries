// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "test_moe_flatmm_base.hpp"
#include "test_moe_flatmm_scenarios.hpp"

// BF16 MoE FlatMM on CDNA. These 2-byte configs are shared by gfx942/gfx950.
// clang-format off
using BF16Types = ::testing::Types<
    std::tuple<BF16, BF16, BF16, FlatmmConfig16<BF16>, GateOnly>,
    std::tuple<BF16, BF16, BF16, FlatmmConfig16<BF16>, GateUp>,
    std::tuple<BF16, BF16, BF16, FlatmmConfig16<BF16>, Gemm2>,
    std::tuple<BF16, BF16, BF16, FlatmmConfig32<BF16>, GateOnly>,
    std::tuple<BF16, BF16, BF16, FlatmmConfig32<BF16>, GateUp>,
    std::tuple<BF16, BF16, BF16, FlatmmConfig32<BF16>, Gemm2>>;
// clang-format on

template <typename Tuple>
class TestMoeFlatmmBF16 : public TestMoeFlatmmBase<Tuple>
{
};

TYPED_TEST_SUITE(TestMoeFlatmmBF16, BF16Types);

MOE_FLATMM_DECLARE_SCENARIOS(TestMoeFlatmmBF16)

// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "test_moe_flatmm_base.hpp"
#include "test_moe_flatmm_scenarios.hpp"

// FP16 MoE FlatMM on CDNA. These 2-byte configs are shared by gfx942/gfx950.
// clang-format off
using FP16Types = ::testing::Types<
    std::tuple<FP16, FP16, FP16, FlatmmConfig16<FP16>, GateOnly>,
    std::tuple<FP16, FP16, FP16, FlatmmConfig16<FP16>, GateUp>,
    std::tuple<FP16, FP16, FP16, FlatmmConfig16<FP16>, Gemm2>,
    std::tuple<FP16, FP16, FP16, FlatmmConfig32<FP16>, GateOnly>,
    std::tuple<FP16, FP16, FP16, FlatmmConfig32<FP16>, GateUp>,
    std::tuple<FP16, FP16, FP16, FlatmmConfig32<FP16>, Gemm2>>;
// clang-format on

template <typename Tuple>
class TestMoeFlatmmFP16 : public TestMoeFlatmmBase<Tuple>
{
};

TYPED_TEST_SUITE(TestMoeFlatmmFP16, FP16Types);

MOE_FLATMM_DECLARE_SCENARIOS(TestMoeFlatmmFP16)

using FP16SplitKTypes =
    ::testing::Types<std::tuple<FP16, FP16, FP16, FlatmmConfig16<FP16>, SplitK>>;

template <typename Tuple>
class TestMoeFlatmmFP16SplitK : public TestMoeFlatmmBase<Tuple>
{
};

TYPED_TEST_SUITE(TestMoeFlatmmFP16SplitK, FP16SplitKTypes);

TYPED_TEST(TestMoeFlatmmFP16SplitK, Typical)
{
    this->run_test(/*num_tokens=*/64,
                   /*topk=*/2,
                   /*experts=*/8,
                   /*N=*/256,
                   /*K=*/512,
                   /*forced_topk_ids=*/std::nullopt,
                   /*skip_experts_with_zero_token=*/true,
                   /*seed=*/42,
                   /*k_batch=*/2);
}

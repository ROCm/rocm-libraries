// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/host.hpp"

#include <gtest/gtest.h>
#include <memory>

#include "test_gemm_quant_fixtures.hpp"
#include "test_gemm_quant_fusedaquant_fixtures.hpp"

using RowMajor    = ck_tile::tensor_layout::gemm::RowMajor;
using ColumnMajor = ck_tile::tensor_layout::gemm::ColumnMajor;
using FP8         = ck_tile::fp8_t;
using BF16        = ck_tile::bf16_t;
using Half        = ck_tile::half_t;
using ABQuantGrouped =
    std::integral_constant<ck_tile::QuantType, ck_tile::QuantType::ABQuantGrouped>;
using GroupSize       = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
using GroupSize2D128N = ck_tile::QuantGroupShape<ck_tile::sequence<1, 128, 128>>;

// clang-format off
using FusedAQuantStandaloneTypes = ::testing::Types<
    // ColumnMajor A is not supported, test only RowMajor
    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF16, FP8, float, Half, ABQuantGrouped, GemmConfigFuseAQuant, GroupSize, GroupSize, ColumnMajor>,
    std::tuple<RowMajor, RowMajor, RowMajor, RowMajor, BF16, FP8, float, Half, ABQuantGrouped, GemmConfigFuseAQuant, GroupSize, GroupSize, ColumnMajor>,

    std::tuple<RowMajor, ColumnMajor, RowMajor, RowMajor, BF16, FP8, float, Half, ABQuantGrouped, GemmConfigFuseAQuantTransposeC, GroupSize, GroupSize2D128N, ColumnMajor>,
    std::tuple<RowMajor, RowMajor, RowMajor, RowMajor, BF16, FP8, float, Half, ABQuantGrouped, GemmConfigFuseAQuantTransposeC, GroupSize, GroupSize2D128N, ColumnMajor>
>;
// clang-format on

TYPED_TEST_SUITE(TestCkTileGemmFusedAQuantStandalone, FusedAQuantStandaloneTypes);

TYPED_TEST(TestCkTileGemmFusedAQuantStandalone, ABQuantGroupedStandaloneFusedAQuant)
{ this->run_test_with_validation(1024, 1024, 1024); }

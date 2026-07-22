// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_quant_common.hpp"

using BlockScaleTypesQuick = ::testing::Types<std::tuple<RowMajor,
                                                         ColumnMajor,
                                                         RowMajor,
                                                         ColumnMajor,
                                                         FP16,
                                                         FP16,
                                                         E8M0,
                                                         FP16,
                                                         BQuantGrouped,
                                                         GemmConfigMx,
                                                         GroupSize1D_64>,
                                              std::tuple<RowMajor,
                                                         ColumnMajor,
                                                         RowMajor,
                                                         ColumnMajor,
                                                         FP16,
                                                         FP8,
                                                         E8M0,
                                                         FP16,
                                                         BQuantGrouped,
                                                         GemmConfigMx,
                                                         GroupSize1D_64>,
                                              std::tuple<RowMajor,
                                                         RowMajor,
                                                         RowMajor,
                                                         ColumnMajor,
                                                         BF16,
                                                         BF8,
                                                         E8M0,
                                                         BF16,
                                                         BQuantGrouped,
                                                         GemmConfigMx,
                                                         GroupSize1D_64>>;

TYPED_TEST_SUITE(TestCkTileGemmBQuant, BlockScaleTypesQuick);

TYPED_TEST(TestCkTileGemmBQuant, BlockScaleMx) { this->run_test_with_validation(256, 256, 256); }

using PreshuffleTypesQuick = ::testing::Types<std::tuple<RowMajor,
                                                         ColumnMajor,
                                                         RowMajor,
                                                         ColumnMajor,
                                                         FP8,
                                                         FP8,
                                                         float,
                                                         Half,
                                                         BQuantGrouped,
                                                         GemmConfigPreshuffleBDecode,
                                                         GroupSize1D_128>,
                                              std::tuple<RowMajor,
                                                         ColumnMajor,
                                                         RowMajor,
                                                         ColumnMajor,
                                                         FP8,
                                                         FP8,
                                                         float,
                                                         Half,
                                                         BQuantGrouped,
                                                         GemmConfigPreshuffleBPrefill,
                                                         GroupSize1D_128>>;

TYPED_TEST_SUITE(TestCkTileGemmPreshuffleBBQuant, PreshuffleTypesQuick);

TYPED_TEST(TestCkTileGemmPreshuffleBBQuant, PreshuffleB)
{
    this->run_test_with_validation(256, 256, 256);
}

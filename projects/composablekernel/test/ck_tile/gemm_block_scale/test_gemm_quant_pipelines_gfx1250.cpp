// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_quant_common.hpp"

// Every gfx1250 GEMM pipeline (mem, compv3, compv4, comp_async, comp_tdm v1/v2) for each quant
// type. Tuple format: <ALayout, BLayout, CLayout, AQLayout, ADataType, BDataType, QDataType,
// CDataType, QuantType, GemmConfig, QuantGroupSize>
template <typename QuantT, typename Config>
using Fp8Rcr = std::tuple<RowMajor,
                          ColumnMajor,
                          RowMajor,
                          RowMajor,
                          FP8,
                          FP8,
                          float,
                          Half,
                          QuantT,
                          Config,
                          GroupSize1D_128>;

template <typename QuantT>
using AllPipelines = ::testing::Types<Fp8Rcr<QuantT, GemmConfigMem>,
                                      Fp8Rcr<QuantT, GemmConfigCompV3>,
                                      Fp8Rcr<QuantT, GemmConfigCompV4>,
                                      Fp8Rcr<QuantT, GemmConfigCompAsync>,
                                      Fp8Rcr<QuantT, GemmConfigCompTDMV1>,
                                      Fp8Rcr<QuantT, GemmConfigCompTDMV2>>;

TYPED_TEST_SUITE(TestCkTileGemmRowColQuant, AllPipelines<RowColQuant>);
TYPED_TEST(TestCkTileGemmRowColQuant, AllPipelines)
{
    this->run_test_with_validation(1024, 1024, 1024);
}

TYPED_TEST_SUITE(TestCkTileGemmTensorQuant, AllPipelines<TensorQuant>);
TYPED_TEST(TestCkTileGemmTensorQuant, AllPipelines)
{
    this->run_test_with_validation(1024, 1024, 1024);
}

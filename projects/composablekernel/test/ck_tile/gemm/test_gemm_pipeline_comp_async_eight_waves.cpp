// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_util.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileGemmPipelineCompAsyncEightWaves
    : public TestCkTileGemmPipeline<T, TestCkTileGemmPipelineCompAsyncEightWaves<T>>
{
    public:
    static constexpr bool check_data_type() { return true; }
    static constexpr bool check_data_shape([[maybe_unused]] ck_tile::index_t M,
                                           [[maybe_unused]] ck_tile::index_t N,
                                           [[maybe_unused]] ck_tile::index_t K,
                                           [[maybe_unused]] bool padM,
                                           [[maybe_unused]] bool padN,
                                           [[maybe_unused]] bool padK)
    {
        return true;
    }
};

#define TEST_SUITE_NAME TestCkTileGemmPipelineCompAsyncEightWaves

TYPED_TEST_SUITE(TEST_SUITE_NAME, KernelTypesCompAsyncEightWaves);

#include "test_gemm_pipeline_ut_cases.inc"

#undef TEST_SUITE_NAME

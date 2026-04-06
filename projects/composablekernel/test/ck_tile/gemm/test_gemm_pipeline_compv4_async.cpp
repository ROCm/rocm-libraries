// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_util.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileGemmPipelineCompV4Async
    : public TestCkTileGemmPipeline<T, TestCkTileGemmPipelineCompV4Async<T>>
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
        return (!padK ||
                (K % TestCkTileGemmPipeline<T, TestCkTileGemmPipelineCompV4Async>::K_Tile) == 0);
    }
};

#define TEST_SUITE_NAME TestCkTileGemmPipelineCompV4Async

TYPED_TEST_SUITE(TEST_SUITE_NAME, KernelTypesCompV4Async);

#include "test_gemm_pipeline_ut_cases.inc"

#undef TEST_SUITE_NAME

template <typename T>
class TestCkTileGemmPipelineCompV4Async16x16x128
    : public TestCkTileGemmPipeline<T, TestCkTileGemmPipelineCompV4Async16x16x128<T>>
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

TYPED_TEST_SUITE(TestCkTileGemmPipelineCompV4Async16x16x128, KernelTypesCompAsync16x16x128);
TYPED_TEST(TestCkTileGemmPipelineCompV4Async16x16x128, QuickTest)
{
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;

    this->template RunSingle<false, false, false, false>(M, N, K, 0, 0, 0, 1);
}

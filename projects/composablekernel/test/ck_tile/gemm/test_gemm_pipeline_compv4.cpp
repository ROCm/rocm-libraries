// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_util.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileGemmPipelineCompV4
    : public TestCkTileGemmPipeline<T, TestCkTileGemmPipelineCompV4<T>>
{
    public:
    static constexpr bool check_data_type()
    {
        using Base = TestCkTileGemmPipeline<T, TestCkTileGemmPipelineCompV4<T>>;
        if constexpr(std::is_same_v<typename Base::BDataType, I4>)
        {
            return false;
        }
        return true;
    }

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

#define TEST_SUITE_NAME TestCkTileGemmPipelineCompV4

TYPED_TEST_SUITE(TEST_SUITE_NAME, KernelTypesCompV4);

#include "test_gemm_pipeline_ut_cases.inc"

#undef TEST_SUITE_NAME

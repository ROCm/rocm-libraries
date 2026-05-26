// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_util.hpp"
#include "gtest/gtest.h"

template <typename T>
class TestCkTileGemmPipelineCompV3PackTensor
    : public TestCkTileGemmPipeline<T, TestCkTileGemmPipelineCompV3PackTensor<T>>
{
    public:
    using Base = TestCkTileGemmPipeline<T, TestCkTileGemmPipelineCompV3PackTensor<T>>;

    static constexpr bool check_data_type()
    {
        if constexpr(std::is_same_v<typename Base::BLayout, Row> &&
                     std::is_same_v<typename Base::BDataType, I4>)
        {
            return false;
        }

        return true;
    }
};

#define TEST_SUITE_NAME TestCkTileGemmPipelineCompV3PackTensor

TYPED_TEST_SUITE(TEST_SUITE_NAME, KernelTypesCompV3WmmaPackTensor);

TYPED_TEST(TEST_SUITE_NAME, SingleTile)
{
    this->Run(TestFixture::M_Tile, TestFixture::N_Tile, TestFixture::K_Tile);
}

TYPED_TEST(TEST_SUITE_NAME, Regular) { this->Run(512, 1024, 512); }

#undef TEST_SUITE_NAME

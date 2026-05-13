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
                                           ck_tile::index_t K,
                                           [[maybe_unused]] bool padM,
                                           [[maybe_unused]] bool padN,
                                           bool padK)
    {
        // The merged CompV4 pipeline routes Row,Col layouts through the async device path
        // on gfx950. async_load_tile cannot mask out-of-bounds K elements, so padK with a
        // K not divisible by K_Tile is unsupported there. Skip those shapes to match the
        // original CompV4Async test's check_data_shape.
        using Base = TestCkTileGemmPipeline<T, TestCkTileGemmPipelineCompV4<T>>;
        if constexpr(std::is_same_v<typename Base::ALayout,
                                    ck_tile::tensor_layout::gemm::RowMajor> &&
                     std::is_same_v<typename Base::BLayout,
                                    ck_tile::tensor_layout::gemm::ColumnMajor>)
        {
            return (!padK || (K % Base::K_Tile) == 0);
        }
        return true;
    }
};

#define TEST_SUITE_NAME TestCkTileGemmPipelineCompV4

TYPED_TEST_SUITE(TEST_SUITE_NAME, KernelTypesCompV4);

#include "test_gemm_pipeline_ut_cases.inc"

#undef TEST_SUITE_NAME

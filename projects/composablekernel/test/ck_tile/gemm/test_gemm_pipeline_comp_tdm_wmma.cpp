// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_wmma_base.hpp"
#include "gtest/gtest.h"

namespace {

template <ck_tile::index_t KPerBlock>
struct TdmPaddingProblem
{
    using ADataType = ck_tile::fp16_t;
    using BDataType = ck_tile::fp16_t;
    using ALayout   = ck_tile::tensor_layout::gemm::RowMajor;
    using BLayout   = ck_tile::tensor_layout::gemm::ColumnMajor;

    struct BlockGemmShape
    {
        static constexpr ck_tile::index_t kM = 128;
        static constexpr ck_tile::index_t kN = 128;
        static constexpr ck_tile::index_t kK = KPerBlock;
    };
};

using TdmPolicy = ck_tile::GemmPipelineAgBgCrCompTDMDefaultPolicy<true>;

template <ck_tile::index_t KPerBlock>
constexpr bool is_tdm_padding_enabled()
{
    constexpr auto a_config = TdmPolicy::GetLdsPaddingConfig<TdmPaddingProblem<KPerBlock>, true>();
    constexpr auto b_config = TdmPolicy::GetLdsPaddingConfig<TdmPaddingProblem<KPerBlock>, false>();
    return a_config[ck_tile::number<0>{}] && b_config[ck_tile::number<0>{}];
}

static_assert(is_tdm_padding_enabled<32>());
static_assert(!is_tdm_padding_enabled<96>());
static_assert(is_tdm_padding_enabled<128>());
static_assert(!is_tdm_padding_enabled<192>());

} // namespace

template <typename T>
class TestCkTileGemmPipelineCompTDMWmma
    : public TestCkTileGemmPipelineWmmaBase<T, TestCkTileGemmPipelineCompTDMWmma<T>>
{
};

#define TEST_SUITE_NAME TestCkTileGemmPipelineCompTDMWmma

TYPED_TEST_SUITE(TestCkTileGemmPipelineCompTDMWmma, KernelTypesCompTDMWmma);

#include "test_gemm_pipeline_ut_cases.inc"

#undef TEST_SUITE_NAME

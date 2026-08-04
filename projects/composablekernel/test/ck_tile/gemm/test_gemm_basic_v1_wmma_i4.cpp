// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "test_gemm_pipeline_wmma_base.hpp"

namespace {

static_assert(ck_tile::has_wmma_traits_v<ck_tile::gfx11_t,
                                         ck_tile::pk_int4_t,
                                         ck_tile::pk_int4_t,
                                         ck_tile::int32_t,
                                         16,
                                         16,
                                         16>);
static_assert(!ck_tile::has_wmma_traits_v<ck_tile::gfx120_t,
                                          ck_tile::pk_int4_t,
                                          ck_tile::pk_int4_t,
                                          ck_tile::int32_t,
                                          16,
                                          16,
                                          16>);

template <ck_tile::GemmPipelineScheduler Scheduler>
using NativeI4Config =
    std::tuple<ck_tile::tensor_layout::gemm::RowMajor,
               ck_tile::tensor_layout::gemm::ColumnMajor,
               ck_tile::tensor_layout::gemm::RowMajor,
               ck_tile::pk_int4_t,
               ck_tile::pk_int4_t,
               ck_tile::int32_t,
               ck_tile::int32_t,
               ck_tile::number<16>,
               ck_tile::number<128>,
               ck_tile::number<128>,
               ck_tile::number<16>,
               ck_tile::number<16>,
               ck_tile::integral_constant<ck_tile::GemmPipelineScheduler, Scheduler>,
               ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::BasicV1>,
               std::false_type,
               std::false_type,
               ck_tile::number<1>,
               ck_tile::number<8>,
               ck_tile::number<1>,
               std::true_type,
               ck_tile::number<1>>;

class TestCkTileGemmBasicV1WmmaI4Intrawave
    : public TestCkTileGemmPipelineWmmaBase<
          NativeI4Config<ck_tile::GemmPipelineScheduler::Intrawave>,
          TestCkTileGemmBasicV1WmmaI4Intrawave>
{
};

class TestCkTileGemmBasicV1WmmaI4Interwave
    : public TestCkTileGemmPipelineWmmaBase<
          NativeI4Config<ck_tile::GemmPipelineScheduler::Interwave>,
          TestCkTileGemmBasicV1WmmaI4Interwave>
{
};

} // namespace

TEST_F(TestCkTileGemmBasicV1WmmaI4Intrawave, ExactSignedPackedValues)
{
    this->Run(16, 128, 128);
    this->Run(5, 127, 128, 144, 144, 129);
}

TEST_F(TestCkTileGemmBasicV1WmmaI4Interwave, ExactSignedPackedValues)
{
    this->Run(16, 128, 128);
    this->Run(5, 127, 128, 144, 144, 129);
}

TEST_F(TestCkTileGemmBasicV1WmmaI4Intrawave, RejectsUnsupportedKVectorTail)
{
    EXPECT_THROW(this->Run(5, 127, 130, 144, 144, 129), std::runtime_error);
}

TEST_F(TestCkTileGemmBasicV1WmmaI4Interwave, RejectsUnsupportedKVectorTail)
{
    EXPECT_THROW(this->Run(5, 127, 130, 144, 144, 129), std::runtime_error);
}

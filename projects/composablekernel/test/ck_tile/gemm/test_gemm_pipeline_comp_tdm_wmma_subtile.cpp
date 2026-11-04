// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// comp_tdm / comp_tdm_v2 at 4 waves (2x2x1) with tiles whose
// VGPR estimate forces sub_tile_num > 1 in the TDM policy.
#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_wmma_base.hpp"
#include "gtest/gtest.h"

#include <string>

// clang-format off
using KernelTypesCompTDMSubTile = ::testing::Types<
    // sub_tile_num = 2: the block GEMM hot loop must cover one K sub-tile only
    std::tuple<Row, Col, Row, F16,  F16,  F32, F16, I128, I128, I256, I16, I16, Intrawave, CompTDMV1>,
    std::tuple<Row, Col, Row, F16,  F16,  F32, F16, I128, I128, I256, I16, I16, Intrawave, CompTDMV2>,
    std::tuple<Row, Col, Row, BF16, BF16, F32, F16, I128, I128, I256, I16, I16, Intrawave, CompTDMV1>,
    std::tuple<Row, Col, Row, BF16, BF16, F32, F16, I128, I128, I256, I16, I16, Intrawave, CompTDMV2>,
    std::tuple<Row, Col, Row, F16,  F16,  F32, F16, I128, I256, I128, I16, I16, Intrawave, CompTDMV1>,
    std::tuple<Row, Col, Row, BF16, BF16, F32, F16, I128, I256, I128, I16, I16, Intrawave, CompTDMV2>,
    // sub_tile_num = 1 controls
    std::tuple<Row, Col, Row, F16,  F16,  F32, F16, I128, I128, I128, I16, I16, Intrawave, CompTDMV1>,
    std::tuple<Row, Col, Row, BF16, BF16, F32, F16, I128, I128, I128, I16, I16, Intrawave, CompTDMV2>
>;
// clang-format on

template <typename T>
class TestCkTileGemmPipelineCompTDMSubTile
    : public TestCkTileGemmPipelineWmmaBase<T, TestCkTileGemmPipelineCompTDMSubTile<T>>
{
    protected:
    // These instances use the regular (non-cluster) launch, which also works on
    // asicRevision=0, so the base-class TDM revision skip does not apply here.
    void SetUp() override { this->k_batches_ = {1}; }
};

TYPED_TEST_SUITE(TestCkTileGemmPipelineCompTDMSubTile, KernelTypesCompTDMSubTile);

TYPED_TEST(TestCkTileGemmPipelineCompTDMSubTile, Regular) { this->Run(512, 1024, 512); }
TYPED_TEST(TestCkTileGemmPipelineCompTDMSubTile, LargeMatrix) { this->Run(2048, 2048, 2048); }
TYPED_TEST(TestCkTileGemmPipelineCompTDMSubTile, Llama) { this->Run(5608, 4096, 2048); }

// K steps that 2 does not divide. The old policy drops the tail; 128x128x96 stays at 1 sub-tile.
using I96  = ck_tile::number<96>;
using I160 = ck_tile::number<160>;

// clang-format off
using SubTile256x256x96    = std::tuple<Col, Row, Row, BF16, BF16, F32, BF16, I256, I256, I96,  I16, I16, Intrawave, CompTDMV1>;
using SubTile128x256x160   = std::tuple<Col, Row, Row, BF16, BF16, F32, BF16, I128, I256, I160, I16, I16, Intrawave, CompTDMV1>;
using SubTile256x256x96K   = std::tuple<Row, Col, Row, BF16, BF16, F32, BF16, I256, I256, I96,  I16, I16, Intrawave, CompTDMV1>;
using SubTile128x128x96    = std::tuple<Col, Row, Row, BF16, BF16, F32, BF16, I128, I128, I96,  I16, I16, Intrawave, CompTDMV1>;
using SubTile256x256x96V2  = std::tuple<Col, Row, Row, BF16, BF16, F32, BF16, I256, I256, I96,  I16, I16, Intrawave, CompTDMV2>;
using SubTile128x256x160V2 = std::tuple<Col, Row, Row, BF16, BF16, F32, BF16, I128, I256, I160, I16, I16, Intrawave, CompTDMV2>;
using SubTile256x256x96KV2 = std::tuple<Row, Col, Row, BF16, BF16, F32, BF16, I256, I256, I96,  I16, I16, Intrawave, CompTDMV2>;
using SubTile128x128x96V2  = std::tuple<Col, Row, Row, BF16, BF16, F32, BF16, I128, I128, I96,  I16, I16, Intrawave, CompTDMV2>;
// clang-format on

template <typename Tuple>
struct SubTileTruncationCase
    : TestCkTileGemmPipelineWmmaBase<Tuple, SubTileTruncationCase<Tuple>>
{
    void TestBody() override {}

    void RunCase()
    {
        using Base = TestCkTileGemmPipelineWmmaBase<Tuple, SubTileTruncationCase<Tuple>>;
        SCOPED_TRACE("tile M=" + std::to_string(Base::M_Tile) + " N=" + std::to_string(Base::N_Tile) +
                     " K=" + std::to_string(Base::K_Tile));
        ASSERT_TRUE(this->check_data_type()) << "gfx1250 bf16 WMMA 16x16x32 is required";
        this->k_batches_ = {1};
        this->Run(Base::M_Tile, Base::N_Tile, Base::K_Tile);
        this->Run(2 * Base::M_Tile, 2 * Base::N_Tile, 4 * Base::K_Tile);
    }
};

// Preferred count is 2, but K/32 is 3 or 5, so the old policy drops the K tail.
TEST(TestCkTileGemmPipelineCompTDMSubTile, SubTileTruncation)
{
    (SubTileTruncationCase<SubTile256x256x96>{}.RunCase(),
     SubTileTruncationCase<SubTile128x256x160>{}.RunCase(),
     SubTileTruncationCase<SubTile256x256x96K>{}.RunCase(),
     SubTileTruncationCase<SubTile128x128x96>{}.RunCase(),
     SubTileTruncationCase<SubTile256x256x96V2>{}.RunCase(),
     SubTileTruncationCase<SubTile128x256x160V2>{}.RunCase(),
     SubTileTruncationCase<SubTile256x256x96KV2>{}.RunCase(),
     SubTileTruncationCase<SubTile128x128x96V2>{}.RunCase());
}

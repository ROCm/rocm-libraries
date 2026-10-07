// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// comp_tdm_ring: TDM ring pipeline of depth 2-4 (gfx1250 only).
#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_wmma_base.hpp"
#include "gtest/gtest.h"

using CompTDMRing = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::CompTDMRing>;
using CompTDMRingD2FillThenWait =
    ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::CompTDMRingD2FillThenWait>;
using CompTDMRingD4PerOperandWave =
    ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::CompTDMRingD4PerOperandWave>;
using CompTDMRingD3SplitBarrier =
    ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::CompTDMRingD3SplitBarrier>;

// clang-format off
using KernelTypesCompTDMRing = ::testing::Types<
    //         ALayout, BLayout, CLayout, ADataType, BDataType, AccDataType, CDataType, M_BlockSize, N_BlockSize, K_BlockSize, M_TileSize, N_TileSize, Scheduler, PipelineType
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I128,        I128,         I64,        I16,        I16, Intrawave, CompTDMRing>,
    std::tuple<    Row,     Col,     Row,      BF16,      BF16,         F32,       F16,        I128,        I128,         I64,        I16,        I16, Intrawave, CompTDMRing, Persistent>,
    std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,        I128,        I128,         I64,        I16,        I16, Intrawave, CompTDMRing>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I128,        I128,         I64,        I16,        I16, Intrawave, CompTDMRingD2FillThenWait>,
    std::tuple<    Row,     Col,     Row,      BF16,      BF16,         F32,       F16,        I128,        I128,         I64,        I16,        I16, Intrawave, CompTDMRingD4PerOperandWave>,
    std::tuple<    Col,     Col,     Row,      BF16,      BF16,         F32,       F16,        I128,        I128,         I64,        I16,        I16, Intrawave, CompTDMRingD4PerOperandWave, Persistent>,
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I128,        I128,         I64,        I16,        I16, Intrawave, CompTDMRingD3SplitBarrier>,
    // sub_tile_num = 2 on 4 waves
    std::tuple<    Row,     Col,     Row,       F16,       F16,         F32,       F16,        I128,        I128,        I256,        I16,        I16, Intrawave, CompTDMRingD2FillThenWait>,
    std::tuple<    Row,     Col,     Row,      BF16,      BF16,         F32,       F16,        I128,        I256,        I128,        I16,        I16, Intrawave, CompTDMRingD3SplitBarrier>
>;
// clang-format on

template <typename T>
class TestCkTileGemmPipelineCompTDMRing
    : public TestCkTileGemmPipelineWmmaBase<T, TestCkTileGemmPipelineCompTDMRing<T>>
{
    protected:
    // Regular (non-cluster) launch, so asicRevision=0 is not skipped; the TDM epilogue has no
    // split-K.
    void SetUp() override { this->k_batches_ = {1}; }
};

TYPED_TEST_SUITE(TestCkTileGemmPipelineCompTDMRing, KernelTypesCompTDMRing);

// num_loop 1..9: covers K shorter than the look-ahead (clamped prologue), every tail length of
// the unrolled ring and at least one full unrolled group.
TYPED_TEST(TestCkTileGemmPipelineCompTDMRing, NumLoop)
{
    for(int num_loop = 1; num_loop <= 9; ++num_loop)
        this->Run(256, 512, num_loop * TestFixture::K_Tile);
}

// K not a multiple of K_Tile: the last tile is partial and the clamped look-ahead re-reads it.
TYPED_TEST(TestCkTileGemmPipelineCompTDMRing, RaggedK)
{
    this->Run(256, 512, 504);
    this->Run(256, 512, 8 * TestFixture::K_Tile + 8);
}

TYPED_TEST(TestCkTileGemmPipelineCompTDMRing, RaggedMN) { this->Run(255, 313, 512); }

TYPED_TEST(TestCkTileGemmPipelineCompTDMRing, Regular) { this->Run(512, 1024, 512); }

TYPED_TEST(TestCkTileGemmPipelineCompTDMRing, LargeMatrix) { this->Run(2048, 2048, 2048); }

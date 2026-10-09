// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Test TDM V1/V2 with non-power-of-two K tiles (TDM padding disabled) and
// power-of-two K tiles (TDM padding enabled).
#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_wmma_base.hpp"
#include "gtest/gtest.h"

using I96 = ck_tile::number<96>;

// clang-format off
using KernelTypesCompTDMKTile = ::testing::Types<
    // non-power-of-two K tile: TDM padding disabled
    std::tuple<Row, Col, Row, F16, F16, F32, F16, I128, I128, I96,  I16, I16, Intrawave, CompTDMV1>,
    std::tuple<Row, Col, Row, F16, F16, F32, F16, I128, I128, I96,  I16, I16, Intrawave, CompTDMV2>,
    std::tuple<Row, Col, Row, F16, F16, F32, F16, I128, I128, I192, I16, I16, Intrawave, CompTDMV1>,
    std::tuple<Row, Col, Row, F16, F16, F32, F16, I128, I128, I192, I16, I16, Intrawave, CompTDMV2>,
    // power-of-two K tile: TDM padding enabled
    std::tuple<Row, Col, Row, F16, F16, F32, F16, I128, I128, I64,  I16, I16, Intrawave, CompTDMV1>,
    std::tuple<Row, Col, Row, F16, F16, F32, F16, I128, I128, I64,  I16, I16, Intrawave, CompTDMV2>,
    std::tuple<Row, Col, Row, F16, F16, F32, F16, I128, I128, I128, I16, I16, Intrawave, CompTDMV1>,
    std::tuple<Row, Col, Row, F16, F16, F32, F16, I128, I128, I128, I16, I16, Intrawave, CompTDMV2>
>;
// clang-format on

template <typename T>
class TestCkTileGemmPipelineCompTDMKTile
    : public TestCkTileGemmPipelineWmmaBase<T, TestCkTileGemmPipelineCompTDMKTile<T>>
{
    protected:
    // Non-cluster launch
    void SetUp() override { this->k_batches_ = {1}; }
};

TYPED_TEST_SUITE(TestCkTileGemmPipelineCompTDMKTile, KernelTypesCompTDMKTile);

// Problem is exactly one block tile (single K loop)
TYPED_TEST(TestCkTileGemmPipelineCompTDMKTile, ExactTile)
{
    this->Run(TestFixture::M_Tile, TestFixture::N_Tile, TestFixture::K_Tile);
}

// Different K-Tile multipliers.
// M=2*M_Tile, N=2*N_Tile is the smallest grid where blocks have non-zero M and N tile offsets.
TYPED_TEST(TestCkTileGemmPipelineCompTDMKTile, KLoops)
{
    for(int k_loops : {2, 3, 5, 16})
    {
        this->Run(2 * TestFixture::M_Tile, 2 * TestFixture::N_Tile, k_loops * TestFixture::K_Tile);
    }
}

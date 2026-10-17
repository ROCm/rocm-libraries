// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// gfx1250 comp_tdm bf16 GEMM tests for the shared LDS pad-interval change.
// Each TEST covers one layout rule:
//   FailureK192               AICK-2609 repro, pad every 64 elements
//   EndOfRow                  K = 64, 128, 256, 512, one pad at the end of the row
//   Every32                   K = 96, 160, 224 (K=96 also on comp_tdm_v2)
//   Every128                  K = 384
//   Every512                  K = 1024 (32x32 block, LDS limit)
//   ColumnMajorA              M = 96, 160, 256, same rules on column-major A
//                             (M=160 also on comp_tdm_v2)
#include "test_gemm_pipeline_kernel_types.hpp"
#include "test_gemm_pipeline_wmma_base.hpp"
#include "gtest/gtest.h"

#include <string>

using I96   = ck_tile::number<96>;
using I160  = ck_tile::number<160>;
using I224  = ck_tile::number<224>;
using I384  = ck_tile::number<384>;
using I512  = ck_tile::number<512>;
using I1024 = ck_tile::number<1024>;

// clang-format off
using TileK192  = std::tuple<Col, Col, Row, BF16, BF16, F32, BF16, I64, I64, I192,  I16, I16, Intrawave, CompTDMV1>;
using TileK64   = std::tuple<Col, Col, Row, BF16, BF16, F32, BF16, I64, I64, I64,   I16, I16, Intrawave, CompTDMV1>;
using TileK128  = std::tuple<Col, Col, Row, BF16, BF16, F32, BF16, I64, I64, I128,  I16, I16, Intrawave, CompTDMV1>;
using TileK256  = std::tuple<Col, Col, Row, BF16, BF16, F32, BF16, I64, I64, I256,  I16, I16, Intrawave, CompTDMV1>;
using TileK512  = std::tuple<Col, Col, Row, BF16, BF16, F32, BF16, I64, I64, I512,  I16, I16, Intrawave, CompTDMV1>;
using TileK96   = std::tuple<Col, Col, Row, BF16, BF16, F32, BF16, I64, I64, I96,   I16, I16, Intrawave, CompTDMV1>;
using TileK96V2 = std::tuple<Col, Col, Row, BF16, BF16, F32, BF16, I64, I64, I96,   I16, I16, Intrawave, CompTDMV2>;
using TileK160  = std::tuple<Col, Col, Row, BF16, BF16, F32, BF16, I64, I64, I160,  I16, I16, Intrawave, CompTDMV1>;
using TileK224  = std::tuple<Col, Col, Row, BF16, BF16, F32, BF16, I64, I64, I224,  I16, I16, Intrawave, CompTDMV1>;
using TileK384  = std::tuple<Col, Col, Row, BF16, BF16, F32, BF16, I64, I64, I384,  I16, I16, Intrawave, CompTDMV1>;
using TileK1024 = std::tuple<Col, Col, Row, BF16, BF16, F32, BF16, I32, I32, I1024, I16, I16, Intrawave, CompTDMV1>;
using TileM96   = std::tuple<Col, Col, Row, BF16, BF16, F32, BF16, I96,  I64, I64,  I16, I16, Intrawave, CompTDMV1>;
using TileM160  = std::tuple<Col, Col, Row, BF16, BF16, F32, BF16, I160, I64, I64,  I16, I16, Intrawave, CompTDMV1>;
using TileM160V2= std::tuple<Col, Col, Row, BF16, BF16, F32, BF16, I160, I64, I64,  I16, I16, Intrawave, CompTDMV2>;
using TileM256  = std::tuple<Col, Col, Row, BF16, BF16, F32, BF16, I256, I64, I64,  I16, I16, Intrawave, CompTDMV1>;
// clang-format on

template <typename Tuple>
struct TdmTileCase : TestCkTileGemmPipelineWmmaBase<Tuple, TdmTileCase<Tuple>>
{
    void TestBody() override {}

    void RunCase()
    {
        using Base = TestCkTileGemmPipelineWmmaBase<Tuple, TdmTileCase<Tuple>>;
        SCOPED_TRACE("tile M=" + std::to_string(Base::M_Tile) + " N=" + std::to_string(Base::N_Tile) +
                     " K=" + std::to_string(Base::K_Tile));
        ASSERT_TRUE(this->check_data_type()) << "gfx1250 bf16 WMMA 16x16x32 is required";
        this->k_batches_ = {1};
        if constexpr(Base::M_Tile == 64 && Base::N_Tile == 64 && Base::K_Tile == 192)
        {
            this->Run(120, 10, 40);
        }
        this->Run(Base::M_Tile + 8, 40, Base::K_Tile * 2);
    }
};

template <typename... Tuples>
void RunTiles()
{
    (TdmTileCase<Tuples>{}.RunCase(), ...);
}

// AICK-2609: contiguous K=192, including the reported M=120 N=10 K=40 shape.
TEST(TestCkTileGemmPipelineCompTDMLdsPadding, FailureK192)
{
    RunTiles<TileK192>();
}

// Power-of-two rows whose legacy interval already lands on the row boundary.
TEST(TestCkTileGemmPipelineCompTDMLdsPadding, EndOfRow)
{
    RunTiles<TileK64, TileK128, TileK256, TileK512>();
}

// Largest hardware interval that divides the row: 16 dwords = 32 bf16.
TEST(TestCkTileGemmPipelineCompTDMLdsPadding, Every32)
{
    RunTiles<TileK96, TileK96V2, TileK160, TileK224>();
}

// 64 dwords = 128 bf16.
TEST(TestCkTileGemmPipelineCompTDMLdsPadding, Every128) { RunTiles<TileK384>(); }

// Legacy field does not fit; interval clamps to 256 dwords = 512 bf16.
TEST(TestCkTileGemmPipelineCompTDMLdsPadding, Every512) { RunTiles<TileK1024>(); }

// Column-major A uses the transpose descriptor with the same interval rules.
TEST(TestCkTileGemmPipelineCompTDMLdsPadding, ColumnMajorA)
{
    RunTiles<TileM96, TileM160, TileM160V2, TileM256>();
}

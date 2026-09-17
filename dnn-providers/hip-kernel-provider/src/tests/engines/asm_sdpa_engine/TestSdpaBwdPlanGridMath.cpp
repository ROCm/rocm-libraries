// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "engines/asm_sdpa_engine/plans/SdpaBwdParams.hpp"

using asm_sdpa_engine::SdpaBwdParams;

namespace
{

// POC tile sizes sourced from
// asm_kernels/gfx942/fmha_v3_bwd/fmha_bwd_{odo,dqdkdv,dq_convert}.csv for the
// hd128/bf16/a32/rtne/psskddv configuration ('ts' column):
//   bwd_hd128_odo_bf16              -> ts=128
//   bwd_hd128_bf16_a32_rtne_psskddv -> ts=192
//   bwd_hd128_dq_convert_bf16_rtne  -> ts=64
constexpr SdpaBwdParams::KernelTiles K_POC_ODO_TILES{128U};
constexpr SdpaBwdParams::KernelTiles K_POC_DQDKDV_TILES{192U};
constexpr SdpaBwdParams::KernelTiles K_POC_DQ_CONVERT_TILES{64U};

// Historical hard-coded grid-tile constants that lived in SdpaBwdPlan.cpp prior
// to CSV-driven plumbing.  Pinned here to make the regression check explicit.
constexpr unsigned int K_HISTORICAL_TS_ODO = 128U;
constexpr unsigned int K_HISTORICAL_TS_KV = 192U;
constexpr unsigned int K_HISTORICAL_TS_DQ = 64U;

SdpaBwdParams makePocParams(unsigned int seqLenQ, unsigned int seqLenKv)
{
    SdpaBwdParams p{};
    p.batchSize = 2U;
    p.numHeadsQ = 16U;
    p.numHeadsKv = 16U;
    p.seqLenQ = seqLenQ;
    p.seqLenKv = seqLenKv;
    p.headDimQk = 128U;
    p.headDimV = 128U;
    p.odoTiles = K_POC_ODO_TILES;
    p.dqdkdvTiles = K_POC_DQDKDV_TILES;
    p.dqConvertTiles = K_POC_DQ_CONVERT_TILES;
    return p;
}

} // namespace

// =============================================================================
// KernelTiles::gridDim — direct unit tests
// =============================================================================

TEST(TestSdpaBwdKernelTiles, GridDimExactMultipleProducesNoRemainder)
{
    constexpr SdpaBwdParams::KernelTiles TILES{64U};
    EXPECT_EQ(TILES.gridDim(64U), 1U);
    EXPECT_EQ(TILES.gridDim(128U), 2U);
    EXPECT_EQ(TILES.gridDim(2048U), 32U);
}

TEST(TestSdpaBwdKernelTiles, GridDimRoundsUpForPartialTile)
{
    constexpr SdpaBwdParams::KernelTiles TILES{192U};
    EXPECT_EQ(TILES.gridDim(1U), 1U);
    EXPECT_EQ(TILES.gridDim(191U), 1U);
    EXPECT_EQ(TILES.gridDim(193U), 2U);
}

TEST(TestSdpaBwdKernelTiles, GridDimZeroExtentReturnsZero)
{
    constexpr SdpaBwdParams::KernelTiles TILES{128U};
    EXPECT_EQ(TILES.gridDim(0U), 0U);
}

TEST(TestSdpaBwdKernelTiles, GridDimZeroTileReturnsZeroInsteadOfDividingByZero)
{
    constexpr SdpaBwdParams::KernelTiles TILES{};
    EXPECT_EQ(TILES.gridDim(2048U), 0U);
}

// =============================================================================
// POC regression — grid-x must match the values produced by the deleted
// K_TS_ODO/K_TS_KV/K_TS_DQ constants for the original hd128/bf16/a32/rtne POC
// configuration on a representative seqlen.
// =============================================================================

TEST(TestSdpaBwdPlanGridMath, PocHd128Bf16Seq2048MatchesHistoricalConstants)
{
    constexpr unsigned int K_SEQ_LEN_Q = 2048U;
    constexpr unsigned int K_SEQ_LEN_KV = 2048U;
    const auto params = makePocParams(K_SEQ_LEN_Q, K_SEQ_LEN_KV);

    const unsigned int historicalGdxOdo
        = (K_SEQ_LEN_Q + K_HISTORICAL_TS_ODO - 1U) / K_HISTORICAL_TS_ODO;
    const unsigned int historicalGdxDqdkdv
        = (K_SEQ_LEN_KV + K_HISTORICAL_TS_KV - 1U) / K_HISTORICAL_TS_KV;
    const unsigned int historicalGdxPost
        = (K_SEQ_LEN_Q + K_HISTORICAL_TS_DQ - 1U) / K_HISTORICAL_TS_DQ;

    EXPECT_EQ(params.odoTiles.gridDim(params.seqLenQ), historicalGdxOdo);
    EXPECT_EQ(params.dqdkdvTiles.gridDim(params.seqLenKv), historicalGdxDqdkdv);
    EXPECT_EQ(params.dqConvertTiles.gridDim(params.seqLenQ), historicalGdxPost);
}

TEST(TestSdpaBwdPlanGridMath, PocHd128Bf16Seq2048MatchesCsvDerivedExpectedDims)
{
    // 2048 / 128 = 16 odo TILES
    // 2048 / 192 = ceil(10.66) = 11 dqdkdv TILES
    // 2048 / 64  = 32 dq_convert TILES
    const auto params = makePocParams(2048U, 2048U);
    EXPECT_EQ(params.odoTiles.gridDim(params.seqLenQ), 16U);
    EXPECT_EQ(params.dqdkdvTiles.gridDim(params.seqLenKv), 11U);
    EXPECT_EQ(params.dqConvertTiles.gridDim(params.seqLenQ), 32U);
}

TEST(TestSdpaBwdPlanGridMath, PocHd128Bf16NonMultipleSeqLenStillRoundsUp)
{
    // Pick lengths that are NOT multiples of any tile to stress ceil-div.
    const auto params = makePocParams(513U, 257U);
    // 513 / 128 = ceil(4.0078) = 5
    EXPECT_EQ(params.odoTiles.gridDim(params.seqLenQ), 5U);
    // 257 / 192 = ceil(1.34) = 2
    EXPECT_EQ(params.dqdkdvTiles.gridDim(params.seqLenKv), 2U);
    // 513 / 64 = ceil(8.015) = 9
    EXPECT_EQ(params.dqConvertTiles.gridDim(params.seqLenQ), 9U);
}

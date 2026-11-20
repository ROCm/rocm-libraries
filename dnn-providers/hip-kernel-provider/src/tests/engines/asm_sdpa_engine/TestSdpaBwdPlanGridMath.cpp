// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "engines/asm_sdpa_engine/plans/SdpaBwdLaunchParams.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaBwdParams.hpp"

#include <cstdint>

using asm_sdpa_engine::computeBwdLaunchParams;
using asm_sdpa_engine::SdpaBwdLaunchParams;
using asm_sdpa_engine::SdpaBwdParams;
using asm_sdpa_engine::SdpaBwdStage;

namespace
{

// Tile sizes ('ts' column) of the gfx942 hd128/bf16/a32/rtne/psskddv kernels in
// asm_kernels/gfx942/fmha_v3_bwd/fmha_bwd_{odo,dqdkdv,dq_convert}.csv:
//   bwd_hd128_odo_bf16              -> ts=128
//   bwd_hd128_bf16_a32_rtne_psskddv -> ts=192
//   bwd_hd128_dq_convert_bf16_rtne  -> ts=64
constexpr SdpaBwdParams::KernelTiles K_HD128_ODO_TILES{128U};
constexpr SdpaBwdParams::KernelTiles K_HD128_DQDKDV_TILES{192U};
constexpr SdpaBwdParams::KernelTiles K_HD128_DQ_CONVERT_TILES{64U};

SdpaBwdParams makeHd128Params(unsigned int seqLenQ, unsigned int seqLenKv, int32_t maskOrdinal)
{
    SdpaBwdParams p{};
    p.batchSize = 2U;
    p.numHeadsQ = 16U;
    p.numHeadsKv = 16U;
    p.seqLenQ = seqLenQ;
    p.seqLenKv = seqLenKv;
    p.headDimQk = 128U;
    p.headDimV = 128U;
    p.odoTiles = K_HD128_ODO_TILES;
    p.dqdkdvTiles = K_HD128_DQDKDV_TILES;
    p.postTiles = K_HD128_DQ_CONVERT_TILES;
    p.archString = "gfx942";
    p.maskOrdinal = maskOrdinal;
    return p;
}

// `what` names the stage and the expected gridDimX in failure messages.
void expectLaunch(const char* what,
                  const SdpaBwdLaunchParams& actual,
                  const SdpaBwdLaunchParams& expected)
{
    SCOPED_TRACE(what);
    EXPECT_EQ(actual.gridDimX, expected.gridDimX);
    EXPECT_EQ(actual.gridDimY, expected.gridDimY);
    EXPECT_EQ(actual.gridDimZ, expected.gridDimZ);
    EXPECT_EQ(actual.blockDimX, expected.blockDimX);
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
// computeBwdLaunchParams — grid and block per stage
// =============================================================================

// Without a causal mask every stage runs one workgroup per tile: ODO and post
// over S_q, DQDKDV over S_kv. The lengths are multiples of no tile size, and
// S_q != S_kv, so a stage that tiles the wrong extent or rounds down fails.
TEST(TestSdpaBwdPlanGridMath, NonCausalLaunchesOneWorkgroupPerTile)
{
    const auto params = makeHd128Params(513U, 257U, 0);

    expectLaunch("ODO: ceil(513 / 128)",
                 computeBwdLaunchParams(params, SdpaBwdStage::ODO),
                 {5U, 16U, 2U, 256U});
    expectLaunch("DQDKDV: ceil(257 / 192)",
                 computeBwdLaunchParams(params, SdpaBwdStage::DQDKDV),
                 {2U, 16U, 2U, 256U});
    expectLaunch("post: ceil(513 / 64)",
                 computeBwdLaunchParams(params, SdpaBwdStage::POST),
                 {9U, 16U, 2U, 256U});
}

// The causal DQDKDV kernels run on half the K/V tiles, rounded up; ODO and
// post keep one workgroup per Q tile.
TEST(TestSdpaBwdPlanGridMath, CausalHalvesOnlyTheDqdkdvGrid)
{
    for(const int32_t mask : {1, 2})
    {
        SCOPED_TRACE(mask);
        const auto params = makeHd128Params(2048U, 2048U, mask);

        expectLaunch("ODO: 2048 / 128",
                     computeBwdLaunchParams(params, SdpaBwdStage::ODO),
                     {16U, 16U, 2U, 256U});
        expectLaunch("DQDKDV: ceil(ceil(2048 / 192) / 2)",
                     computeBwdLaunchParams(params, SdpaBwdStage::DQDKDV),
                     {6U, 16U, 2U, 256U});
        expectLaunch("post: 2048 / 64",
                     computeBwdLaunchParams(params, SdpaBwdStage::POST),
                     {32U, 16U, 2U, 256U});
    }
}

// =============================================================================
// DQDKDV grid halving: only the causal kernels tile half the K/V blocks
// =============================================================================

// AITER v0.1.29 mha_bwd.cu fmha_v3_bwd() halves gdx for masks 1 and 2 only.
// The sliding-window kernels (mask 3, and mask 4 on gfx950) and the unmasked
// kernels take one workgroup per K/V tile.
TEST(TestSdpaBwdPlanGridMath, OnlyCausalMasksHalveTheDqdkdvGrid)
{
    for(const int32_t mask : {1, 2})
    {
        SCOPED_TRACE(mask);
        EXPECT_TRUE(asm_sdpa_engine::halvesDqdkdvGrid(mask));
    }
    for(const int32_t mask : {0, 3, 4})
    {
        SCOPED_TRACE(mask);
        EXPECT_FALSE(asm_sdpa_engine::halvesDqdkdvGrid(mask));
    }
}

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "engines/asm_sdpa_engine/plans/SdpaFwdLaunchParams.hpp"

using asm_sdpa_engine::computeFwdLaunchParams;
using asm_sdpa_engine::SdpaFwdLaunchParams;
using asm_sdpa_engine::SdpaFwdParams;
using asm_sdpa_engine::plan_utils::MaskType;

namespace
{

// Standard hd128 configuration (the most common SDPA shape).
SdpaFwdParams makeHd128Params()
{
    SdpaFwdParams p{};
    p.batchSize = 2U;
    p.numHeadsQ = 16U;
    p.numHeadsKv = 16U;
    p.seqLenQ = 2048U;
    p.seqLenKv = 2048U;
    p.headDimQk = 128U;
    p.headDimV = 128U;
    p.tileSizeQo = 256U;
    p.archString = "gfx942";
    p.maskType = MaskType::NO_MASK;
    return p;
}

// hd192x128 / gfx942 variant — triggers the swap + blockDim=256 path.
SdpaFwdParams makeHd192x128Params()
{
    SdpaFwdParams p{};
    p.batchSize = 2U;
    p.numHeadsQ = 16U;
    p.numHeadsKv = 16U;
    p.seqLenQ = 2048U;
    p.seqLenKv = 2048U;
    p.headDimQk = 192U;
    p.headDimV = 128U;
    p.tileSizeQo = 128U;
    p.archString = "gfx942";
    p.maskType = MaskType::NO_MASK;
    return p;
}

} // namespace

// =============================================================================
// Grid dimension tests
// =============================================================================

TEST(TestSdpaFwdPlanGridMath, NoMaskGridDimXMatchesCeilDiv)
{
    auto p = makeHd128Params();
    // 2048 / 256 = 8, no halving
    auto lp = computeFwdLaunchParams(p);
    EXPECT_EQ(lp.gridDimX, 8U);
    EXPECT_EQ(lp.gridDimY, 16U);
    EXPECT_EQ(lp.gridDimZ, 2U);
}

TEST(TestSdpaFwdPlanGridMath, NoMaskGridDimXRoundsUpPartialTile)
{
    auto p = makeHd128Params();
    p.seqLenQ = 257U;
    // ceil(257 / 256) = 2
    auto lp = computeFwdLaunchParams(p);
    EXPECT_EQ(lp.gridDimX, 2U);
}

TEST(TestSdpaFwdPlanGridMath, NoMaskZeroTileSizeReturnsZero)
{
    auto p = makeHd128Params();
    p.tileSizeQo = 0U;
    auto lp = computeFwdLaunchParams(p);
    EXPECT_EQ(lp.gridDimX, 0U);
    EXPECT_EQ(lp.gridDimY, 0U);
    EXPECT_EQ(lp.gridDimZ, 0U);
}

TEST(TestSdpaFwdPlanGridMath, CausalMaskHalvesGridDimX)
{
    auto p = makeHd128Params();
    p.maskType = MaskType::TOP_LEFT_CAUSAL;
    // 2048/256 = 8, halved to 4
    auto lp = computeFwdLaunchParams(p);
    EXPECT_EQ(lp.gridDimX, 4U);
    EXPECT_EQ(lp.gridDimY, 16U);
}

TEST(TestSdpaFwdPlanGridMath, BottomRightCausalAlsoHalvesGridDimX)
{
    auto p = makeHd128Params();
    p.maskType = MaskType::BOTTOM_RIGHT_CAUSAL;
    auto lp = computeFwdLaunchParams(p);
    EXPECT_EQ(lp.gridDimX, 4U);
}

TEST(TestSdpaFwdPlanGridMath, CausalOddTileCountTruncates)
{
    auto p = makeHd128Params();
    p.maskType = MaskType::TOP_LEFT_CAUSAL;
    p.seqLenQ = 768U;
    // ceil(768/256) = 3, 3/2 = 1 (integer truncation)
    auto lp = computeFwdLaunchParams(p);
    EXPECT_EQ(lp.gridDimX, 1U);
}

TEST(TestSdpaFwdPlanGridMath, WindowGenericMaskAlsoHalvesGridDimX)
{
    auto p = makeHd128Params();
    p.maskType = MaskType::SLIDING_WINDOW;
    // ordinal 3 is also != 0, so tg_div applies
    auto lp = computeFwdLaunchParams(p);
    EXPECT_EQ(lp.gridDimX, 4U);
}

TEST(TestSdpaFwdPlanGridMath, Hd192x128SwapsGridDimXY)
{
    auto p = makeHd192x128Params();
    // 2048/128 = 16 tiles, no mask → no halving, then swap
    // After swap: X=numHeadsQ=16, Y=16 (tiles)
    auto lp = computeFwdLaunchParams(p);
    EXPECT_EQ(lp.gridDimX, 16U); // was gridDimY (numHeadsQ) before swap
    EXPECT_EQ(lp.gridDimY, 16U); // was gridDimX (tiles) before swap
    EXPECT_EQ(lp.blockDimX, 256U);
}

TEST(TestSdpaFwdPlanGridMath, Hd192x128CausalHalvesThenSwaps)
{
    auto p = makeHd192x128Params();
    p.maskType = MaskType::TOP_LEFT_CAUSAL;
    // tiles = 2048/128 = 16, halved to 8, then swap X/Y
    auto lp = computeFwdLaunchParams(p);
    // After swap: X=numHeadsQ=16, Y=8 (halved tiles)
    EXPECT_EQ(lp.gridDimX, 16U);
    EXPECT_EQ(lp.gridDimY, 8U);
}

// =============================================================================
// tune_opt tests
// =============================================================================

TEST(TestSdpaFwdPlanTuneOpt, NoMaskAlwaysReturns5)
{
    auto p = makeHd128Params();
    auto lp = computeFwdLaunchParams(p);
    EXPECT_EQ(lp.tuneOpt, 5U);
}

TEST(TestSdpaFwdPlanTuneOpt, NoMaskReturns5EvenWithNonAlignedHeads)
{
    auto p = makeHd128Params();
    p.numHeadsQ = 7U; // not 8-aligned, but no mask → no downgrade
    auto lp = computeFwdLaunchParams(p);
    EXPECT_EQ(lp.tuneOpt, 5U);
}

TEST(TestSdpaFwdPlanTuneOpt, CausalWith8AlignedHeadsAndShortSeqReturns5)
{
    auto p = makeHd128Params();
    p.maskType = MaskType::TOP_LEFT_CAUSAL;
    p.numHeadsQ = 8U;
    p.seqLenQ = 4096U; // <= 16384
    auto lp = computeFwdLaunchParams(p);
    EXPECT_EQ(lp.tuneOpt, 5U);
}

TEST(TestSdpaFwdPlanTuneOpt, CausalWithNonAlignedHeadsDowngradesTo3)
{
    auto p = makeHd128Params();
    p.maskType = MaskType::TOP_LEFT_CAUSAL;
    p.numHeadsQ = 7U; // not 8-aligned → downgrade
    auto lp = computeFwdLaunchParams(p);
    EXPECT_EQ(lp.tuneOpt, 3U);
}

TEST(TestSdpaFwdPlanTuneOpt, CausalWithSeqLenOver16KDowngradesTo3)
{
    auto p = makeHd128Params();
    p.maskType = MaskType::TOP_LEFT_CAUSAL;
    p.numHeadsQ = 16U; // 8-aligned
    p.seqLenQ = 16385U; // > 16384 → downgrade
    auto lp = computeFwdLaunchParams(p);
    EXPECT_EQ(lp.tuneOpt, 3U);
}

TEST(TestSdpaFwdPlanTuneOpt, CausalWithSeqLenExactly16KReturns5)
{
    auto p = makeHd128Params();
    p.maskType = MaskType::TOP_LEFT_CAUSAL;
    p.numHeadsQ = 16U;
    p.seqLenQ = 16384U; // boundary: 16384 is not > 16384
    auto lp = computeFwdLaunchParams(p);
    EXPECT_EQ(lp.tuneOpt, 5U);
}

TEST(TestSdpaFwdPlanTuneOpt, Hd192x128Gfx942AlwaysReturns0)
{
    auto p = makeHd192x128Params();
    auto lp = computeFwdLaunchParams(p);
    EXPECT_EQ(lp.tuneOpt, 0U);
}

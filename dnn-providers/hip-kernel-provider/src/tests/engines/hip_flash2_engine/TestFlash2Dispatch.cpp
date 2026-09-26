// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Unit tests for the Flash2 shape -> kernel-variant dispatch rule.
//
// These are pure host-side tests: selectFlash2Config() is a arithmetic
// function over the problem shape, so no device is required and the cases
// run everywhere.
//
// The expected values are the variants actually measured fastest for each
// shape in the 21-shape MI300X sweep. They are a regression fence: if the
// rule is retuned, these must be re-measured, not simply updated to match
// whatever the new rule emits.

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <string>

#include "engines/hip_flash2_engine/Flash2Dispatch.hpp"

namespace hip_flash2_engine
{
namespace
{

struct DispatchCase
{
    int batch;
    int heads;
    int seqLen;
    int headDim;
    bool causal;
    const char* wantTag;
    int wantSplitK;
};

std::string describe(const DispatchCase& testCase)
{
    return "B" + std::to_string(testCase.batch) + " H" + std::to_string(testCase.heads) + " S"
           + std::to_string(testCase.seqLen) + " D" + std::to_string(testCase.headDim)
           + (testCase.causal ? " causal" : " noncausal");
}

// Measured-best variant per shape, from the 21-shape MI300X sweep.
const std::array<DispatchCase, 21> K_MEASURED_CASES{{
    {1, 32, 512, 128, true, "w4q1k4", 1},
    {1, 32, 1024, 128, true, "w8q1k4", 1},
    {1, 32, 2048, 128, true, "w8q2k4", 1},
    {1, 32, 4096, 128, true, "w8q3k2", 1},
    {1, 32, 2048, 128, false, "w8q2k4", 1},
    {1, 32, 4096, 128, false, "w8q2k4", 1},
    {1, 32, 2048, 64, true, "w8q2k4", 1},
    {4, 32, 1024, 128, true, "w8q3k2", 1},
    {1, 32, 8192, 128, true, "w8q3k2", 1},
    {1, 32, 8192, 128, false, "w8q3k2", 1},
    {2, 16, 3072, 128, false, "w8q3k2", 1},
    {1, 64, 1536, 64, true, "w8q2k4", 1},
    {8, 32, 512, 128, true, "w8q2k4", 1},
    {1, 16, 6144, 128, true, "w8q3k2", 1},
    {1, 32, 3072, 128, true, "w8q3k2", 1},
    {2, 8, 8192, 128, true, "w8q3k2", 1},
    {16, 32, 256, 128, true, "w8q2k4", 1},
    // Starved grid. The measured-best pairing is w8q2k4 WITH splitK=4 (191
    // vs 78 TFLOPS single-pass), but execute() does not run the split yet, so
    // the rule must not credit it: 64 CTAs stay 64 and that is a tiny grid.
    // When K_FLASH2_SPLITK_EXECUTES flips to true this becomes w8q2k4 again.
    {1, 8, 2048, 128, false, "w4q1k4", 4},
    {2, 32, 2048, 128, true, "w8q3k2", 1},
    {1, 40, 1024, 128, false, "w8q2k4", 1},
    {4, 16, 4096, 64, true, "w8q3k4", 1},
}};

TEST(TestFlash2Dispatch, ReproducesMeasuredVariantSelection)
{
    for(const auto& testCase : K_MEASURED_CASES)
    {
        const Flash2Selection sel = selectFlash2Config(
            testCase.batch, testCase.heads, testCase.seqLen, testCase.headDim, testCase.causal);
        EXPECT_STREQ(sel.variant.tag, testCase.wantTag) << describe(testCase);
        EXPECT_EQ(sel.splitK, testCase.wantSplitK) << describe(testCase);
    }
}

// A variant's launch geometry must match what its .co was compiled with:
// launching a 512-thread object with 64 threads silently computes wrong
// results, and the reverse fails with hipErrorLaunchFailure (719).
TEST(TestFlash2Dispatch, SelectedGeometryIsLaunchable)
{
    for(const auto& testCase : K_MEASURED_CASES)
    {
        const Flash2Selection sel = selectFlash2Config(
            testCase.batch, testCase.heads, testCase.seqLen, testCase.headDim, testCase.causal);
        EXPECT_EQ(sel.variant.blockDim % 64U, 0U) << describe(testCase);
        EXPECT_LE(sel.variant.blockDim, 1024U) << describe(testCase);
        EXPECT_GT(sel.variant.qPerCta, 0U) << describe(testCase);
        // queries/CTA = waves * queryGroups * 16, so it is always a multiple of 16.
        EXPECT_EQ(sel.variant.qPerCta % 16U, 0U) << describe(testCase);
        EXPECT_GE(sel.splitK, 1) << describe(testCase);
    }
}

TEST(TestFlash2Dispatch, WorkspaceIsZeroWithoutSplitK)
{
    EXPECT_EQ(flash2WorkspaceBytes(1, 32, 2048, 128, 1), 0U);
    EXPECT_EQ(flash2WorkspaceBytes(1, 32, 2048, 128, 0), 0U);
}

// Layout: fp32 partial O [B, H, splitK, Sq, D], then per-split m and l.
TEST(TestFlash2Dispatch, WorkspaceSizeMatchesPartialLayout)
{
    constexpr size_t K_ROWS = static_cast<size_t>(1) * 8 * 4 * 2048;
    constexpr size_t K_EXPECT = K_ROWS * 128 * sizeof(float) + 2 * K_ROWS * sizeof(float);
    EXPECT_EQ(flash2WorkspaceBytes(1, 8, 2048, 128, 4), K_EXPECT);
}

// The rule keys on CTA count versus CU count, so the same shape can select a
// different variant on a smaller part. Guard the scaling path explicitly.
TEST(TestFlash2Dispatch, ThresholdsScaleWithCuCount)
{
    // 304-CU part: this shape fills the GPU and takes the general variant.
    const Flash2Selection big = selectFlash2Config(8, 32, 512, 128, true, 304);
    EXPECT_STREQ(big.variant.tag, "w8q2k4");

    // Same shape on a much smaller part still produces a valid, launchable
    // selection -- the thresholds scale rather than falling off a cliff.
    const Flash2Selection small = selectFlash2Config(8, 32, 512, 128, true, 64);
    EXPECT_EQ(small.variant.blockDim % 64U, 0U);
    EXPECT_GE(small.splitK, 1);
}

// Regression: the split-K branch used to `return` immediately, which pinned
// every starved shape to w8q2k4 and suppressed the tiny-grid rule below it.
// splitK and the variant are independent decisions -- a shape can be both
// starved enough to split AND small enough to want the 4-wave variant.
TEST(TestFlash2Dispatch, SplitKDoesNotSuppressTinyGridVariant)
{
    // ct256 = 4 CTAs: far below the tiny-grid threshold of 100.
    const Flash2Selection tiny = selectFlash2Config(1, 1, 1024, 128, false);
    EXPECT_STREQ(tiny.variant.tag, "w4q1k4");
    EXPECT_GT(tiny.splitK, 1) << "starved grid should still split";

    // ct256 = 64. The split factor is still computed and recorded, but while
    // K_FLASH2_SPLITK_EXECUTES is false the tiny-grid rule judges the grid
    // that actually runs -- 64 CTAs, which is tiny -- so the 4-wave variant
    // wins. Crediting the unexecuted split here would pick an 8-wave variant
    // for a grid that stays starved, which is what S. Reeder measured across
    // 90 shapes in the ctas256 [32, 96] band.
    const Flash2Selection split = selectFlash2Config(1, 8, 2048, 128, false);
    EXPECT_EQ(split.splitK, 4) << "split factor should still be recorded";
    if(K_FLASH2_SPLITK_EXECUTES)
    {
        EXPECT_STREQ(split.variant.tag, "w8q2k4");
    }
    else
    {
        EXPECT_STREQ(split.variant.tag, "w4q1k4");
    }

    // ct256 = 128: fills the GPU, so neither rule fires.
    const Flash2Selection full = selectFlash2Config(1, 8, 4096, 128, false);
    EXPECT_EQ(full.splitK, 1);
}

// The selection rule must describe what the engine does, not what it will do.
// If split-K is wired through execute(), flip K_FLASH2_SPLITK_EXECUTES in the
// same change -- this test exists so that cannot be forgotten in either
// direction.
TEST(TestFlash2Dispatch, SplitKCreditMatchesExecutionReality)
{
    EXPECT_FALSE(K_FLASH2_SPLITK_EXECUTES)
        << "execute() still forces splitK=1; if that changed, update this test "
           "and the tiny-grid expectations above together";
}

TEST(TestFlash2Dispatch, SplitKOnlyForLongD128Sequences)
{
    // head_dim 64 has no split-K kernel.
    EXPECT_EQ(selectFlash2Config(1, 2, 2048, 64, false).splitK, 1);
    // Short sequences are not worth the merge pass.
    EXPECT_EQ(selectFlash2Config(1, 2, 512, 128, false).splitK, 1);
    // Split factor is clamped to [2, 4]; 8 never won on any measured shape.
    EXPECT_LE(selectFlash2Config(1, 1, 2048, 128, false).splitK, 4);
}

TEST(TestFlash2Dispatch, CtaCountAccountsForBatchHeadsAndSequence)
{
    // ceil(2048/256) * 2 * 16 = 8 * 32 = 256
    EXPECT_EQ(flash2CtaCount(2, 16, 2048, 256), 256);
    // Partial tiles round up.
    EXPECT_EQ(flash2CtaCount(1, 1, 257, 256), 2);
    EXPECT_EQ(flash2CtaCount(1, 1, 256, 256), 1);
}

} // namespace
} // namespace hip_flash2_engine

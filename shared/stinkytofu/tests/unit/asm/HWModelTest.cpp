/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
// Pins the CDNA5 (Gfx1250) hardware-model facts.
//
// These assertions read the real HWModel through hwModelForArch(), so a change to
// any migrated constant fails here. Scheduling *behaviour* built on top of these
// numbers is covered by DAGSchedulerPassTest.cpp and tests/filecheck/dag_*.stir.
#include <gtest/gtest.h>

#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/HWModel.hpp"
#include "stinkytofu/transforms/asm/dag/HazardRules.hpp"

using namespace stinkytofu;

namespace {
constexpr std::array<int, 3> kGfx1250 = {12, 5, 0};
constexpr std::array<int, 3> kGfx1250v0 = {12, 5, 1};
}  // namespace

TEST(HWModel, Gfx1250KnownDefaults) {
    const HWModel& hw = hwModelForArch(kGfx1250);

    EXPECT_EQ(hw.lds.readQueueDepth, 16);
    // 0 means "use dynamic drain latency model", not a fixed drain value.
    EXPECT_EQ(hw.lds.readDrainLatency, 0);
    EXPECT_EQ(hw.lds.readThrottleLatency, 72);

    EXPECT_EQ(hw.barrier.signalToWaitLatency, 11);
    EXPECT_EQ(hw.barrier.jumpOverheadCycles, 6);

    EXPECT_EQ(hw.coexec.transToNonCoreSide, 1);
    EXPECT_EQ(hw.coexec.maxSlotBudget, 18);
}

TEST(HWModel, Gfx1250HazardRules) {
    const HWModel& hw = hwModelForArch(kGfx1250);

    ASSERT_EQ(hw.hazards.numRules, kNumCdna5HazardRules);
    ASSERT_EQ(hw.hazards.numRules, 4);
    ASSERT_NE(hw.hazards.rules, nullptr);

    // The model points at the shared table rather than carrying a copy.
    EXPECT_EQ(hw.hazards.rules, kCdna5HazardRules);

    EXPECT_STREQ(hw.hazards.rules[0].name, "SaluSgprToMemAddr");
    EXPECT_EQ(hw.hazards.rules[0].regType, RegType::S);
    EXPECT_EQ(hw.hazards.rules[0].distance, 8);
    EXPECT_EQ(hw.hazards.rules[0].dir, HazardDir::WriteThenRead);
    EXPECT_EQ(hw.hazards.rules[0].unit, HazardUnit::Cycles);

    EXPECT_STREQ(hw.hazards.rules[1].name, "ValuVgprToVmemAddr");
    EXPECT_EQ(hw.hazards.rules[1].regType, RegType::V);
    EXPECT_EQ(hw.hazards.rules[1].distance, 32);
    EXPECT_EQ(hw.hazards.rules[1].dir, HazardDir::WriteThenRead);
    EXPECT_EQ(hw.hazards.rules[1].unit, HazardUnit::Cycles);

    // Same pair as rule[1]; -1 means "hoist as far as possible", so it contributes a
    // deadline of 0 and no gate. rule[1] remains the one that enforces the gap.
    EXPECT_STREQ(hw.hazards.rules[2].name, "ValuVgprToVmemAddrHoist");
    EXPECT_EQ(hw.hazards.rules[2].regType, RegType::V);
    EXPECT_EQ(hw.hazards.rules[2].distance, -1);
    EXPECT_EQ(hw.hazards.rules[2].dir, HazardDir::WriteThenRead);
    EXPECT_EQ(hw.hazards.rules[2].unit, HazardUnit::Cycles);

    EXPECT_STREQ(hw.hazards.rules[3].name, "WmmaVgprSrcToDsWrite");
    EXPECT_EQ(hw.hazards.rules[3].regType, RegType::V);
    EXPECT_EQ(hw.hazards.rules[3].distance, 0);
    EXPECT_EQ(hw.hazards.rules[3].dir, HazardDir::ReadThenWrite);
    EXPECT_EQ(hw.hazards.rules[3].unit, HazardUnit::PipeOps);
    EXPECT_NE(hw.hazards.rules[3].isPipeOp, nullptr);
}

// gfx1250v0 currently aliases gfx1250 field-for-field. This pins that it is a
// deliberate alias of the same values, not an accidental fallthrough: when
// gfx1250v0 gets its own tuning, this test is the one that should be updated.
TEST(HWModel, Gfx1250v0MatchesGfx1250ForNow) {
    const HWModel& base = hwModelForArch(kGfx1250);
    const HWModel& v0 = hwModelForArch(kGfx1250v0);

    EXPECT_EQ(v0.lds.readQueueDepth, base.lds.readQueueDepth);
    EXPECT_EQ(v0.lds.readDrainLatency, base.lds.readDrainLatency);
    EXPECT_EQ(v0.lds.readThrottleLatency, base.lds.readThrottleLatency);
    EXPECT_EQ(v0.barrier.signalToWaitLatency, base.barrier.signalToWaitLatency);
    EXPECT_EQ(v0.hazards.rules, base.hazards.rules);
}

// An unlisted arch falls back to gfx1250 — and must return the *same object*, so
// callers that cache the reference stay valid.
TEST(HWModel, UnlistedArchFallsBackToGfx1250) {
    EXPECT_EQ(&hwModelForArch({9, 4, 2}), &hwModelForArch(kGfx1250));
}

// Many unit tests construct a bare PassContext, and EstimateAsmCyclesPass builds
// one internally; none of those call setGemmTileConfig. getHWModel() must still
// answer rather than dereference a null cached pointer.
TEST(HWModel, BarePassContextReturnsDefaultModel) {
    PassContext ctx;
    EXPECT_EQ(&ctx.getHWModel(), &hwModelForArch(kGfx1250));
}

TEST(HWModel, ConfiguredPassContextCachesMatchingModel) {
    GemmTileConfig cfg;
    cfg.arch = kGfx1250;
    PassContext ctx;
    ctx.setGemmTileConfig(cfg);
    EXPECT_EQ(&ctx.getHWModel(), &hwModelForArch(kGfx1250));
}

// Pins the wait-hide counts. Behaviour built on them is covered by
// tests/filecheck/InsertWaitAluPass_xdl_hide_test.stir and _vmvsrc_hide_test.stir.
TEST(HWModel, Gfx1250WaitHideCounts) {
    const HWModel& hw = hwModelForArch(kGfx1250);
    EXPECT_EQ(hw.waitHide.xdlVaVdst, 12);
    EXPECT_EQ(hw.waitHide.csmaccVaVdst, 13);
    EXPECT_EQ(hw.waitHide.vmVsrc, 11);

    const HWModel& v0 = hwModelForArch(kGfx1250v0);
    EXPECT_EQ(v0.waitHide.xdlVaVdst, hw.waitHide.xdlVaVdst);
    EXPECT_EQ(v0.waitHide.csmaccVaVdst, hw.waitHide.csmaccVaVdst);
    EXPECT_EQ(v0.waitHide.vmVsrc, hw.waitHide.vmVsrc);
}

// The zero-is-off convention. No arch currently ships a 0, so this is the only
// coverage of the inert path an arch opting out would take.
TEST(HWModel, WaitHideSatisfiedZeroIsOff) {
    EXPECT_FALSE(waitHideSatisfied(/*count=*/1000, /*step=*/1, /*required=*/0));
    EXPECT_FALSE(waitHideSatisfied(1000, 2, 0));
    EXPECT_FALSE(waitHideSatisfied(1000, 1, -1));
}

// Boundary: satisfied at exactly the count, not one below. \p step scales the
// requirement, so a 2-unit step needs twice the raw count.
TEST(HWModel, WaitHideSatisfiedBoundaryAndStep) {
    EXPECT_FALSE(waitHideSatisfied(11, 1, 12));
    EXPECT_TRUE(waitHideSatisfied(12, 1, 12));
    EXPECT_TRUE(waitHideSatisfied(13, 1, 12));

    EXPECT_FALSE(waitHideSatisfied(23, 2, 12));
    EXPECT_TRUE(waitHideSatisfied(24, 2, 12));
}

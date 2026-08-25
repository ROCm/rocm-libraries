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
    EXPECT_EQ(hw.lds.readDrainLatency, 72);
    EXPECT_EQ(hw.lds.readThrottleLatency, 72);

    EXPECT_EQ(hw.barrier.signalToWaitLatency, 11);
    EXPECT_EQ(hw.barrier.jumpOverheadCycles, 6);

    EXPECT_EQ(hw.coexec.transToNonCoreSide, 1);
    EXPECT_EQ(hw.coexec.maxSlotBudget, 18);
}

// Barrier drain latency, exercised with the gfx1250 figures: a 16-deep queue that
// drains in 72 cycles, fed by ds_load_b128 (56 cycles).
namespace {
constexpr int kDepth = 16;
constexpr int kFullQueueDrain = 72;
constexpr int kDsLoad = 56;

int barrierDrain(int matchingDsLoadCount) {
    return computeBarrierDrainLatency(kDepth, kFullQueueDrain, matchingDsLoadCount, kDsLoad);
}
}  // namespace

TEST(HWModel, BarrierDrainLatencyRampsBetweenOneLoadAndAFullQueue) {
    // Nothing is queued behind a lone load, so it costs only its own return.
    EXPECT_EQ(barrierDrain(1), kDsLoad);
    EXPECT_EQ(barrierDrain(kDepth), kFullQueueDrain);

    // Monotone in between, and never outside those two endpoints.
    for (int count = 2; count < kDepth; ++count) {
        EXPECT_GE(barrierDrain(count), barrierDrain(count - 1)) << "count=" << count;
        EXPECT_LE(barrierDrain(count), kFullQueueDrain) << "count=" << count;
    }
    // A count-aware ramp, not a flat budget: 1 and depth must not agree.
    EXPECT_GT(barrierDrain(kDepth), barrierDrain(1));
}

// Regression guard: only a queue's worth of loads can still be outstanding when the
// barrier is reached, so a large matching count must not scale the wait past a full
// drain. Letting it scale pushed the barrier past the end of its region, which cost
// the TDM prefetch all of its overlap with the WMMA stream.
TEST(HWModel, BarrierDrainLatencySaturatesAtQueueDepth) {
    EXPECT_EQ(barrierDrain(kDepth + 1), kFullQueueDrain);
    EXPECT_EQ(barrierDrain(126), kFullQueueDrain);
    EXPECT_EQ(barrierDrain(10000), kFullQueueDrain);
}

// An arch with no modeled LDS return queue has no drain to wait on.
TEST(HWModel, BarrierDrainLatencyWithoutAQueueIsJustTheLoadLatency) {
    EXPECT_EQ(computeBarrierDrainLatency(/*queueDepth=*/0, kFullQueueDrain, /*count=*/8, kDsLoad),
              kDsLoad);
}

TEST(HWModel, Gfx1250HazardRules) {
    const HWModel& hw = hwModelForArch(kGfx1250);

    ASSERT_EQ(hw.hazards.numRules, kNumCdna5HazardRules);
    ASSERT_EQ(hw.hazards.numRules, 2);
    ASSERT_NE(hw.hazards.rules, nullptr);

    // The model points at the shared table rather than carrying a copy.
    EXPECT_EQ(hw.hazards.rules, kCdna5HazardRules);

    EXPECT_STREQ(hw.hazards.rules[0].name, "SaluSgprToMemAddr");
    EXPECT_EQ(hw.hazards.rules[0].regType, RegType::S);
    EXPECT_EQ(hw.hazards.rules[0].cycles, 8);

    EXPECT_STREQ(hw.hazards.rules[1].name, "ValuVgprToVmemAddr");
    EXPECT_EQ(hw.hazards.rules[1].regType, RegType::V);
    EXPECT_EQ(hw.hazards.rules[1].cycles, 32);
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

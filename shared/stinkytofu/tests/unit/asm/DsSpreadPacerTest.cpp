// Copyright (C) 2025-2026 Advanced Micro Devices, Inc.

#include <gtest/gtest.h>

#include <climits>

#include "transforms/asm/dag/DsSpreadPacer.hpp"

using namespace stinkytofu;

namespace {
constexpr int kWindow = 8;  // cycles per WMMA window

std::vector<DsSpreadPacer::Load> loadsWithoutConsumers(int n) {
    std::vector<DsSpreadPacer::Load> loads;
    for (int i = 0; i < n; ++i) loads.push_back({static_cast<unsigned>(i), INT_MAX});
    return loads;
}
}  // namespace

TEST(DsSpreadPacer, StrictSpreadReleasesLoadsAtTheirPlannedCycle) {
    DsSpreadPacer pacer;
    // 16 WMMAs of 8 cycles, 4 loads: one every 32 cycles, planned at 0, 32, 64, 96.
    pacer.plan(loadsWithoutConsumers(4), 16 * kWindow, kWindow, /*tolerance=*/-1, /*lead=*/16);
    EXPECT_EQ(pacer.periodCycles(), 32);
    EXPECT_TRUE(pacer.released(0, 0));
    EXPECT_FALSE(pacer.released(1, 31));
    EXPECT_TRUE(pacer.released(1, 32));
    EXPECT_FALSE(pacer.released(3, 95));
    EXPECT_TRUE(pacer.released(3, 96));
}

TEST(DsSpreadPacer, ABacklogDrainsAtThePlannedRateNotAsABurst) {
    DsSpreadPacer pacer;
    pacer.plan(loadsWithoutConsumers(4), 16 * kWindow, kWindow, -1, 16);
    // Nothing issued until cycle 100: every load is past its plan, but they leave one period apart.
    EXPECT_TRUE(pacer.released(0, 100));
    pacer.noteIssue(100);
    EXPECT_FALSE(pacer.released(1, 101));
    EXPECT_FALSE(pacer.released(1, 131));
    EXPECT_TRUE(pacer.released(1, 132));
    pacer.noteIssue(132);
    EXPECT_FALSE(pacer.released(2, 163));
    EXPECT_TRUE(pacer.released(2, 164));
}

TEST(DsSpreadPacer, ManualToleranceReleasesEarlier) {
    DsSpreadPacer pacer;
    pacer.plan(loadsWithoutConsumers(4), 16 * kWindow, kWindow, /*tolerance=*/16, 16);
    EXPECT_EQ(pacer.toleranceCycles(), 16);
    EXPECT_FALSE(pacer.released(1, 15));
    EXPECT_TRUE(pacer.released(1, 16));
}

TEST(DsSpreadPacer, AutomaticToleranceIsStrict) {
    DsSpreadPacer pacer;
    pacer.plan(loadsWithoutConsumers(4), 16 * kWindow, kWindow, /*tolerance=*/-1, 16);
    EXPECT_EQ(pacer.toleranceCycles(), 0);
}

TEST(DsSpreadPacer, ATightConsumerReleasesOnlyItsOwnLoad) {
    // Load 2 is planned at 64 but its consumer is WMMA 3: with a lead of 16 cycles it must go by
    // 3 * 8 - 16 = 8, whatever the plan and the rate say. The other loads keep their plan.
    std::vector<DsSpreadPacer::Load> loads = {{0, INT_MAX}, {1, INT_MAX}, {2, 3}, {3, INT_MAX}};
    DsSpreadPacer pacer;
    pacer.plan(loads, 16 * kWindow, kWindow, -1, 16);
    pacer.noteIssue(7);
    EXPECT_FALSE(pacer.released(2, 7));
    EXPECT_TRUE(pacer.released(2, 8));
    EXPECT_FALSE(pacer.released(1, 8));
    EXPECT_FALSE(pacer.released(3, 8));
}

TEST(DsSpreadPacer, PlanStartsAtTheRegionClock) {
    DsSpreadPacer pacer;
    pacer.plan(loadsWithoutConsumers(2), 8 * kWindow, kWindow, -1, 16, /*startClock=*/100);
    EXPECT_FALSE(pacer.released(1, 131));
    EXPECT_TRUE(pacer.released(1, 132));
}

TEST(DsSpreadPacer, NoWmmaOrNoLoadsMeansNoHold) {
    DsSpreadPacer pacer;
    pacer.plan(loadsWithoutConsumers(4), 0, kWindow, 0, 16);
    EXPECT_FALSE(pacer.active());
    EXPECT_TRUE(pacer.released(2, 0));
    pacer.plan({}, 16 * kWindow, kWindow, 0, 16);
    EXPECT_FALSE(pacer.active());
}

TEST(DsSpreadPacer, GapCvIsZeroForEvenIssueAndGrowsWithBursts) {
    DsSpreadPacer even;
    for (int c : {0, 8, 16, 24, 32}) even.noteIssue(c);
    EXPECT_DOUBLE_EQ(even.gapCv(), 0.0);

    DsSpreadPacer burst;
    for (int c : {0, 1, 2, 3, 32}) burst.noteIssue(c);
    EXPECT_GT(burst.gapCv(), 1.0);
}

// Copyright (C) 2025-2026 Advanced Micro Devices, Inc.

#include <gtest/gtest.h>

#include <algorithm>
#include <climits>

#include "transforms/asm/dag/DsSpreadPacer.hpp"

using namespace stinkytofu;

namespace {
constexpr int kWindow = 8;        // cycles per WMMA window
constexpr int kTailReserve = 72;  // ds return latency

std::vector<DsSpreadPacer::Load> loadsWithoutConsumers(int n) {
    std::vector<DsSpreadPacer::Load> loads;
    for (int i = 0; i < n; ++i) loads.push_back({static_cast<unsigned>(i), INT_MAX});
    return loads;
}
}  // namespace

TEST(DsSpreadPacer, GapIsTheTimeLeftLessTheTailReserveOverTheLoadsLeft) {
    DsSpreadPacer pacer;
    pacer.plan(loadsWithoutConsumers(4), kWindow, /*tolerance=*/-1, /*lead=*/16, kTailReserve);
    // 16 WMMAs of 8 cycles: (128 - 72) / 4.
    EXPECT_EQ(pacer.gapCycles(128, 4), 14);
    // More time or fewer loads left means a longer gap; less time, a shorter one.
    EXPECT_EQ(pacer.gapCycles(200, 4), 32);
    EXPECT_EQ(pacer.gapCycles(200, 8), 16);
    EXPECT_LT(pacer.gapCycles(100, 4), pacer.gapCycles(128, 4));
}

TEST(DsSpreadPacer, NothingToSpreadOverMeansNoGap) {
    DsSpreadPacer pacer;
    pacer.plan(loadsWithoutConsumers(4), kWindow, -1, 16, kTailReserve);
    EXPECT_EQ(pacer.gapCycles(kTailReserve, 4), 0);  // no time beyond the return latency
    EXPECT_EQ(pacer.gapCycles(0, 4), 0);
    EXPECT_EQ(pacer.gapCycles(128, 0), 0);
}

TEST(DsSpreadPacer, ManualToleranceAddsLeadToTheTailReserve) {
    DsSpreadPacer pacer;
    pacer.plan(loadsWithoutConsumers(4), kWindow, /*tolerance=*/16, 16, kTailReserve);
    EXPECT_EQ(pacer.toleranceCycles(), 16);
    EXPECT_EQ(pacer.gapCycles(128, 4), 10);  // (128 - 72 - 16) / 4
}

TEST(DsSpreadPacer, AutomaticToleranceAddsNoLead) {
    DsSpreadPacer pacer;
    pacer.plan(loadsWithoutConsumers(4), kWindow, /*tolerance=*/-1, 16, kTailReserve);
    EXPECT_EQ(pacer.toleranceCycles(), 0);
}

TEST(DsSpreadPacer, ALoadWaitsOneGapAfterThePreviousLoad) {
    DsSpreadPacer pacer;
    pacer.plan(loadsWithoutConsumers(4), kWindow, -1, 16, kTailReserve);
    EXPECT_TRUE(pacer.released(0, 0, 128, 4));  // nothing issued yet
    pacer.noteIssue(0);
    EXPECT_FALSE(pacer.released(1, 13, 128, 3));  // gap (128 - 72) / 3 = 18
    EXPECT_FALSE(pacer.released(1, 17, 128, 3));
    EXPECT_TRUE(pacer.released(1, 18, 128, 3));
}

TEST(DsSpreadPacer, ASpreadFinishesTheReturnLatencyBeforeTheRegionEndsAndStaysEven) {
    // 24 WMMAs (192 cycles), 12 loads. Drive the clock a cycle at a time as the scheduler would.
    constexpr int kRegion = 24 * kWindow;
    constexpr int kLoads = 12;
    DsSpreadPacer pacer;
    pacer.plan(loadsWithoutConsumers(kLoads), kWindow, -1, 16, kTailReserve);
    std::vector<int> issued;
    for (int clock = 0; clock < kRegion && static_cast<int>(issued.size()) < kLoads; ++clock) {
        const int left = kLoads - static_cast<int>(issued.size());
        if (pacer.released(static_cast<unsigned>(issued.size()), clock, kRegion - clock, left)) {
            pacer.noteIssue(clock);
            issued.push_back(clock);
        }
    }
    ASSERT_EQ(static_cast<int>(issued.size()), kLoads);
    EXPECT_LE(issued.back(), kRegion - kTailReserve);
    for (size_t i = 1; i < issued.size(); ++i) {
        EXPECT_GE(issued[i] - issued[i - 1], 9);
        EXPECT_LE(issued[i] - issued[i - 1], 11);
    }
}

TEST(DsSpreadPacer, ABacklogCatchesUpAtAShorterGapNotAsABurst) {
    DsSpreadPacer pacer;
    pacer.plan(loadsWithoutConsumers(12), kWindow, -1, 16, kTailReserve);
    // Nothing issued for the first 60 cycles of a 192-cycle region: the gap is now shorter than
    // the 10 it started at, but still not zero.
    const int startGap = pacer.gapCycles(192, 12);
    const int lateGap = pacer.gapCycles(192 - 60, 12);
    EXPECT_LT(lateGap, startGap);
    EXPECT_GT(lateGap, 0);
}

TEST(DsSpreadPacer, ATightConsumerReleasesOnlyItsOwnLoad) {
    // Load 2's consumer is WMMA 3: with a lead of 16 cycles it must go by 3 * 8 - 16 = 8, whatever
    // the gap says. The other loads keep the gap.
    std::vector<DsSpreadPacer::Load> loads = {{0, INT_MAX}, {1, INT_MAX}, {2, 3}, {3, INT_MAX}};
    DsSpreadPacer pacer;
    pacer.plan(loads, kWindow, -1, 16, kTailReserve);
    pacer.noteIssue(7);
    EXPECT_FALSE(pacer.released(2, 7, 128, 3));
    EXPECT_TRUE(pacer.released(2, 8, 128, 3));
    EXPECT_FALSE(pacer.released(1, 8, 128, 3));
    EXPECT_FALSE(pacer.released(3, 8, 128, 3));
}

TEST(DsSpreadPacer, DeadlinesStartAtTheRegionClock) {
    std::vector<DsSpreadPacer::Load> loads = {{0, INT_MAX}, {1, 3}};
    DsSpreadPacer pacer;
    pacer.plan(loads, kWindow, -1, 16, kTailReserve, /*startClock=*/100);
    pacer.noteIssue(100);
    EXPECT_FALSE(pacer.released(1, 107, 128, 1));
    EXPECT_TRUE(pacer.released(1, 108, 128, 1));
}

TEST(DsSpreadPacer, NoLoadsMeansNoHold) {
    DsSpreadPacer pacer;
    pacer.plan({}, kWindow, 0, 16, kTailReserve);
    EXPECT_FALSE(pacer.active());
    EXPECT_TRUE(pacer.released(2, 0, 128, 4));
}

TEST(DsSpreadPacer, GapCvIsZeroForEvenIssueAndGrowsWithBursts) {
    DsSpreadPacer even;
    for (int c : {0, 8, 16, 24, 32}) even.noteIssue(c);
    EXPECT_DOUBLE_EQ(even.gapCv(), 0.0);

    DsSpreadPacer burst;
    for (int c : {0, 1, 2, 3, 32}) burst.noteIssue(c);
    EXPECT_GT(burst.gapCv(), 1.0);
}

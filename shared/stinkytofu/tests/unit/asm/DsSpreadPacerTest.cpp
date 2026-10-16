// Copyright (C) 2025-2026 Advanced Micro Devices, Inc.

#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <map>

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
    EXPECT_DOUBLE_EQ(pacer.gapCycles(128, 4), 14.0);
    // More time or fewer loads left means a longer gap; less time, a shorter one.
    EXPECT_DOUBLE_EQ(pacer.gapCycles(200, 4), 32.0);
    EXPECT_DOUBLE_EQ(pacer.gapCycles(200, 8), 16.0);
    EXPECT_LT(pacer.gapCycles(100, 4), pacer.gapCycles(128, 4));
    // The gap is fractional: 120 cycles over 23 loads.
    EXPECT_NEAR(pacer.gapCycles(192, 23), 120.0 / 23, 1e-9);
}

TEST(DsSpreadPacer, NothingToSpreadOverMeansNoGap) {
    DsSpreadPacer pacer;
    pacer.plan(loadsWithoutConsumers(4), kWindow, -1, 16, kTailReserve);
    EXPECT_DOUBLE_EQ(pacer.gapCycles(kTailReserve, 4), 0.0);  // no time beyond the return latency
    EXPECT_DOUBLE_EQ(pacer.gapCycles(0, 4), 0.0);
    EXPECT_DOUBLE_EQ(pacer.gapCycles(128, 0), 0.0);
}

TEST(DsSpreadPacer, ManualToleranceAddsLeadToTheTailReserve) {
    DsSpreadPacer pacer;
    pacer.plan(loadsWithoutConsumers(4), kWindow, /*tolerance=*/16, 16, kTailReserve);
    EXPECT_EQ(pacer.toleranceCycles(), 16);
    EXPECT_DOUBLE_EQ(pacer.gapCycles(128, 4), 10.0);  // (128 - 72 - 16) / 4
}

TEST(DsSpreadPacer, AutomaticToleranceAddsNoLead) {
    DsSpreadPacer pacer;
    pacer.plan(loadsWithoutConsumers(4), kWindow, /*tolerance=*/-1, 16, kTailReserve);
    EXPECT_EQ(pacer.toleranceCycles(), 0);
}

TEST(DsSpreadPacer, ALoadWaitsOneGapAfterThePreviousLoad) {
    DsSpreadPacer pacer;
    pacer.plan(loadsWithoutConsumers(4), kWindow, -1, 16, kTailReserve);
    EXPECT_TRUE(pacer.released(0, 0));  // nothing issued yet
    pacer.noteIssue(0, 128, 4);         // gap (128 - 72) / 4 = 14
    EXPECT_FALSE(pacer.released(1, 13));
    EXPECT_TRUE(pacer.released(1, 14));
}

TEST(DsSpreadPacer, AGapShorterThanAWindowGivesEvenlySpacedDoublesNotATailBurst) {
    // 24 WMMA windows (192 cycles) and 23 loads: the gap is about 5.2 cycles, shorter than the
    // 8-cycle window the clock moves in. Decide once per window, as the scheduler does.
    constexpr int kWindows = 24;
    constexpr int kRegion = kWindows * kWindow;
    constexpr int kLoads = 23;
    DsSpreadPacer pacer;
    pacer.plan(loadsWithoutConsumers(kLoads), kWindow, -1, 16, kTailReserve);
    std::vector<int> issued;
    std::map<int, int> perWindow;
    for (int w = 0; w < kWindows && static_cast<int>(issued.size()) < kLoads; ++w) {
        const int clock = w * kWindow;
        while (static_cast<int>(issued.size()) < kLoads &&
               pacer.released(static_cast<unsigned>(issued.size()), clock)) {
            pacer.noteIssue(clock, kRegion - clock, kLoads - static_cast<int>(issued.size()));
            issued.push_back(clock);
            ++perWindow[w];
        }
    }
    ASSERT_EQ(static_cast<int>(issued.size()), kLoads);
    // The last load goes out about a return latency before the region ends.
    EXPECT_LE(issued.back(), kRegion - kTailReserve + kWindow);
    // Never a burst: one or two loads a window, and the doubles are spread out, not at the end.
    for (const auto& [w, n] : perWindow) EXPECT_LE(n, 2) << "window " << w;
    int doublesInLastThreeIssueWindows = 0;
    std::vector<int> windows;
    for (const auto& [w, n] : perWindow) windows.push_back(w);
    for (size_t i = windows.size() >= 3 ? windows.size() - 3 : 0; i < windows.size(); ++i)
        doublesInLastThreeIssueWindows += perWindow[windows[i]] == 2;
    EXPECT_LE(doublesInLastThreeIssueWindows, 1);
}

TEST(DsSpreadPacer, AStallReleasesOneExtraLoadNotABurst) {
    DsSpreadPacer pacer;
    pacer.plan(loadsWithoutConsumers(12), kWindow, -1, 16, kTailReserve);
    pacer.noteIssue(0, 192, 12);  // gap 10
    // Nothing issues for 100 cycles, then the loads can go: one at the stall's end, one more, and
    // the third waits a gap.
    pacer.noteIssue(100, 92, 11);  // gap (92 - 72) / 11 < 2
    EXPECT_TRUE(pacer.released(2, 100));
    pacer.noteIssue(100, 92, 10);
    EXPECT_FALSE(pacer.released(3, 100));
}

TEST(DsSpreadPacer, ATightConsumerReleasesOnlyItsOwnLoad) {
    // Load 2's consumer is WMMA 3: with a lead of 16 cycles it must go by 3 * 8 - 16 = 8, whatever
    // the gap says. The other loads keep the gap.
    std::vector<DsSpreadPacer::Load> loads = {{0, INT_MAX}, {1, INT_MAX}, {2, 3}, {3, INT_MAX}};
    DsSpreadPacer pacer;
    pacer.plan(loads, kWindow, -1, 16, kTailReserve);
    pacer.noteIssue(7, 128, 4);  // gap 14: next allowed at 21
    EXPECT_FALSE(pacer.released(2, 7));
    EXPECT_TRUE(pacer.released(2, 8));
    EXPECT_FALSE(pacer.released(1, 8));
    EXPECT_FALSE(pacer.released(3, 8));
}

TEST(DsSpreadPacer, DeadlinesStartAtTheRegionClock) {
    std::vector<DsSpreadPacer::Load> loads = {{0, INT_MAX}, {1, 3}};
    DsSpreadPacer pacer;
    pacer.plan(loads, kWindow, -1, 16, kTailReserve, /*startClock=*/100);
    pacer.noteIssue(100, 128, 2);
    EXPECT_FALSE(pacer.released(1, 107));
    EXPECT_TRUE(pacer.released(1, 108));
}

TEST(DsSpreadPacer, NoLoadsMeansNoHold) {
    DsSpreadPacer pacer;
    pacer.plan({}, kWindow, 0, 16, kTailReserve);
    EXPECT_FALSE(pacer.active());
    EXPECT_TRUE(pacer.released(2, 0));
}

TEST(DsSpreadPacer, GapCvIsZeroForEvenIssueAndGrowsWithBursts) {
    DsSpreadPacer even;
    for (int c : {0, 8, 16, 24, 32}) even.noteIssue(c, 0, 0);
    EXPECT_DOUBLE_EQ(even.gapCv(), 0.0);

    DsSpreadPacer burst;
    for (int c : {0, 1, 2, 3, 32}) burst.noteIssue(c, 0, 0);
    EXPECT_GT(burst.gapCv(), 1.0);
}

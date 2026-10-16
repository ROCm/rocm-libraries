// Copyright (C) 2025-2026 Advanced Micro Devices, Inc.

#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <map>

#include "transforms/asm/dag/DsSpreadPacer.hpp"

using namespace stinkytofu;

namespace {
constexpr int kWindow = 8;  // cycles per WMMA window

std::vector<DsSpreadPacer::Load> loadsWithoutConsumers(int n) {
    std::vector<DsSpreadPacer::Load> loads;
    for (int i = 0; i < n; ++i) loads.push_back({static_cast<unsigned>(i), INT_MAX});
    return loads;
}

PassFeatureConfig::DsPace pace(int cycles, int count, int separators) {
    PassFeatureConfig::DsPace p;
    p.cycles = cycles;
    p.count = count;
    p.separators = separators;
    return p;
}
}  // namespace

TEST(DsSpreadPacer, DefaultPaceIsSeparatorsOnly) {
    PassFeatureConfig::DsPaceRegion r;
    r.dsTotal = 4;
    r.separatorTotal = 10;
    const auto p = defaultDsPace(r, /*requestedSeparators=*/-1);
    EXPECT_EQ(p.cycles, 0);
    EXPECT_EQ(p.count, 1);
    EXPECT_EQ(p.separators, 1);
}

TEST(DsSpreadPacer, DefaultPaceGroupsLoadsWhenSeparatorsAreScarce) {
    PassFeatureConfig::DsPaceRegion r;
    r.dsTotal = 30;
    r.separatorTotal = 9;  // 10 groups at most with one separator between them
    EXPECT_EQ(defaultDsPace(r, 1).count, 3);
    // Two separators between groups: 5 groups at most.
    EXPECT_EQ(defaultDsPace(r, 2).count, 6);
    // No separators required: nothing limits the groups.
    EXPECT_EQ(defaultDsPace(r, 0).count, 1);
    EXPECT_EQ(defaultDsPace(r, 0).separators, 0);
}

TEST(DsSpreadPacer, AGroupHoldsCountLoadsBackToBack) {
    DsSpreadPacer pacer;
    pacer.plan(loadsWithoutConsumers(6), pace(10, 2, 1), kWindow, 16);
    EXPECT_TRUE(pacer.released(0, 0));
    pacer.noteIssue(0);
    EXPECT_TRUE(pacer.released(1, 0));  // joins the group, nothing emitted in between
    pacer.noteIssue(0);
    EXPECT_FALSE(pacer.released(2, 0));  // the group is full
}

TEST(DsSpreadPacer, TheNextGroupWaitsForTheCyclesAndForASeparator) {
    DsSpreadPacer pacer;
    pacer.plan(loadsWithoutConsumers(6), pace(10, 2, 1), kWindow, 16);
    pacer.noteIssue(0);
    pacer.noteIssue(0);
    // Time alone is not enough: a ds_load right after a full group would stall on the hardware.
    EXPECT_FALSE(pacer.released(2, 1000));
    pacer.noteSeparator();
    EXPECT_FALSE(pacer.released(2, 9));  // a separator, but too early
    EXPECT_TRUE(pacer.released(2, 10));
}

TEST(DsSpreadPacer, SeparatorsBetweenGroupsAreCounted) {
    DsSpreadPacer pacer;
    pacer.plan(loadsWithoutConsumers(4), pace(0, 1, 2), kWindow, 16);
    pacer.noteIssue(0);
    pacer.noteSeparator();
    EXPECT_FALSE(pacer.released(1, 100));  // one of the two required
    pacer.noteSeparator();
    EXPECT_TRUE(pacer.released(1, 100));
}

TEST(DsSpreadPacer, NoSeparatorsRequiredPacesByCyclesOnly) {
    DsSpreadPacer pacer;
    pacer.plan(loadsWithoutConsumers(4), pace(10, 1, 0), kWindow, 16);
    pacer.noteIssue(0);
    EXPECT_FALSE(pacer.released(1, 9));
    EXPECT_TRUE(pacer.released(1, 10));
}

TEST(DsSpreadPacer, AGroupBrokenBySeparatorsIsANewGroup) {
    DsSpreadPacer pacer;
    pacer.plan(loadsWithoutConsumers(4), pace(10, 3, 1), kWindow, 16);
    pacer.noteIssue(0);
    pacer.noteSeparator();  // something is emitted: the group is over, though it had room
    EXPECT_FALSE(pacer.released(1, 5));
    EXPECT_TRUE(pacer.released(1, 10));
}

TEST(DsSpreadPacer, APeriodShorterThanAWindowDoesNotDriftIntoATailBurst) {
    // One group every 7 cycles with a separator between groups, decided once per 8-cycle window.
    // Counted from the issue every period would round up to a window; counted from the allowed
    // time the 1-cycle surplus per window accumulates.
    DsSpreadPacer pacer;
    pacer.plan(loadsWithoutConsumers(64), pace(7, 1, 1), kWindow, 16);
    int issued = 0;
    for (int w = 0; w < 58; ++w) {
        const int clock = w * kWindow;
        while (issued < 64 && pacer.released(static_cast<unsigned>(issued), clock)) {
            pacer.noteIssue(clock);
            ++issued;
            pacer.noteSeparator();  // a VALU between two groups in the same window
        }
        pacer.noteSeparator();  // the WMMA
    }
    // 64 periods of 7 cycles fit in 448 cycles, 56 windows of 8.
    EXPECT_EQ(issued, 64);
}

TEST(DsSpreadPacer, AStallReleasesOneGroupNotABurst) {
    DsSpreadPacer pacer;
    pacer.plan(loadsWithoutConsumers(12), pace(10, 1, 1), kWindow, 16);
    pacer.noteIssue(0);
    pacer.noteSeparator();
    pacer.noteIssue(100);  // nothing issued for 100 cycles
    pacer.noteSeparator();
    EXPECT_TRUE(pacer.released(2, 100));  // one period behind: this group goes
    pacer.noteIssue(100);
    pacer.noteSeparator();
    EXPECT_FALSE(pacer.released(3, 100));  // the one after waits its period
}

TEST(DsSpreadPacer, ATightConsumerReleasesOnlyItsOwnLoad) {
    // Load 2's consumer is WMMA 3: with a lead of 16 cycles it must go by 3 * 8 - 16 = 8, whatever
    // the pace says. The other loads keep it.
    std::vector<DsSpreadPacer::Load> loads = {{0, INT_MAX}, {1, INT_MAX}, {2, 3}, {3, INT_MAX}};
    DsSpreadPacer pacer;
    pacer.plan(loads, pace(14, 1, 1), kWindow, 16);
    pacer.noteIssue(7);  // next group at 21
    pacer.noteSeparator();
    EXPECT_FALSE(pacer.released(2, 7));
    EXPECT_TRUE(pacer.released(2, 8));
    EXPECT_FALSE(pacer.released(1, 8));
    EXPECT_FALSE(pacer.released(3, 8));
}

TEST(DsSpreadPacer, DeadlinesStartAtTheRegionClock) {
    std::vector<DsSpreadPacer::Load> loads = {{0, INT_MAX}, {1, 3}};
    DsSpreadPacer pacer;
    pacer.plan(loads, pace(50, 1, 1), kWindow, 16, /*startClock=*/100);
    pacer.noteIssue(100);
    pacer.noteSeparator();
    EXPECT_FALSE(pacer.released(1, 107));
    EXPECT_TRUE(pacer.released(1, 108));
}

TEST(DsSpreadPacer, NoLoadsMeansNoHold) {
    DsSpreadPacer pacer;
    pacer.plan({}, pace(10, 1, 1), kWindow, 16);
    EXPECT_FALSE(pacer.active());
    EXPECT_TRUE(pacer.released(2, 0));
}

TEST(DsSpreadPacer, GapCvIsZeroForEvenIssueAndGrowsWithBursts) {
    DsSpreadPacer even;
    for (int c : {0, 8, 16, 24, 32}) even.noteIssue(c);
    EXPECT_DOUBLE_EQ(even.gapCv(), 0.0);

    DsSpreadPacer burst;
    for (int c : {0, 1, 2, 3, 32}) burst.noteIssue(c);
    EXPECT_GT(burst.gapCv(), 1.0);
}

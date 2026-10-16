// Copyright (C) 2025-2026 Advanced Micro Devices, Inc.

#include <gtest/gtest.h>

#include <climits>

#include "transforms/asm/dag/DsSpreadPacer.hpp"

using namespace stinkytofu;

namespace {
std::vector<DsSpreadPacer::Load> loadsWithoutConsumers(int n) {
    std::vector<DsSpreadPacer::Load> loads;
    for (int i = 0; i < n; ++i) loads.push_back({static_cast<unsigned>(i), INT_MAX});
    return loads;
}
}  // namespace

TEST(DsSpreadPacer, StrictSpreadReleasesLoadsAtTheirPlannedWindow) {
    DsSpreadPacer pacer;
    pacer.plan(loadsWithoutConsumers(4), /*wmmaTotal=*/16, /*manualSlack=*/0, /*lead=*/2);
    // Planned windows 0, 4, 8, 12.
    EXPECT_TRUE(pacer.released(0, 0));
    EXPECT_FALSE(pacer.released(1, 3));
    EXPECT_TRUE(pacer.released(1, 4));
    EXPECT_FALSE(pacer.released(3, 11));
    EXPECT_TRUE(pacer.released(3, 12));
}

TEST(DsSpreadPacer, ManualSlackReleasesEarlier) {
    DsSpreadPacer pacer;
    pacer.plan(loadsWithoutConsumers(4), 16, /*manualSlack=*/3, 2);
    EXPECT_TRUE(pacer.released(1, 1));
    EXPECT_FALSE(pacer.released(1, 0));
    EXPECT_EQ(pacer.slack(), 3);
}

TEST(DsSpreadPacer, AutoSlackIsZeroWithoutTightConsumers) {
    DsSpreadPacer pacer;
    pacer.plan(loadsWithoutConsumers(4), 16, /*manualSlack=*/-1, 2);
    EXPECT_EQ(pacer.slack(), 0);
}

TEST(DsSpreadPacer, AutoSlackIsTheSmallestOneThatMeetsEveryConsumer) {
    // Four loads planned at windows 0, 4, 8, 12. The last one is consumed by WMMA 9, and with a
    // lead of 2 it must be released by window 7: 12 - slack <= 7, so slack 5.
    std::vector<DsSpreadPacer::Load> loads = {{0, INT_MAX}, {1, 20}, {2, 20}, {3, 9}};
    DsSpreadPacer pacer;
    pacer.plan(loads, 16, -1, 2);
    EXPECT_EQ(pacer.slack(), 5);
    EXPECT_TRUE(pacer.released(3, 7));
    EXPECT_FALSE(pacer.released(3, 6));
}

TEST(DsSpreadPacer, AutoSlackNeverReleasesEarlierThanNeeded) {
    DsSpreadPacer pacer;
    std::vector<DsSpreadPacer::Load> loads = {{0, 3}, {1, 40}, {2, 40}, {3, 40}};
    pacer.plan(loads, 16, -1, 2);
    // Load 0 is planned at 0 and needed by 3 - 2 = 1: no slack needed.
    EXPECT_EQ(pacer.slack(), 0);
}

TEST(DsSpreadPacer, NoWmmaOrNoLoadsMeansNoHold) {
    DsSpreadPacer pacer;
    pacer.plan(loadsWithoutConsumers(4), 0, 0, 2);
    EXPECT_FALSE(pacer.active());
    EXPECT_TRUE(pacer.released(2, 0));
    pacer.plan({}, 16, 0, 2);
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

TEST(DsSpreadPacer, ExtraGateIsOpenByDefault) {
    DsSpreadPacer pacer;
    EXPECT_TRUE(pacer.extraGate(0));
    pacer.plan(loadsWithoutConsumers(4), /*wmmaTotal=*/8, /*manualSlack=*/0, /*leadWindows=*/2);
    EXPECT_TRUE(pacer.extraGate(100));
}

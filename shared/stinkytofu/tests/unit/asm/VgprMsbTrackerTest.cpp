// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The bank-word state machine InsertVgprMsbPass emits by and the allocator's
// prediction replays.

#include <gtest/gtest.h>

#include "stinkytofu/ir/asm/VgprMsbEncoding.hpp"

using namespace stinkytofu;

namespace {

constexpr int kWordA = 0x5A;
constexpr int kWordB = 0x5B;

}  // namespace

TEST(VgprMsbTrackerTest, AWordIsSetOnceAndKept) {
    VgprMsbTracker tracker;
    const VgprMsbStep first = tracker.step(kWordA, /*hasVgpr=*/true);
    EXPECT_TRUE(first.emit);
    EXPECT_FALSE(first.firstAfterLabel);
    EXPECT_FALSE(first.previousWord.has_value());

    EXPECT_FALSE(tracker.step(kWordA, /*hasVgpr=*/true).emit);

    const VgprMsbStep changed = tracker.step(kWordB, /*hasVgpr=*/true);
    EXPECT_TRUE(changed.emit);
    EXPECT_EQ(changed.previousWord, kWordA);
}

TEST(VgprMsbTrackerTest, AnInstructionWithoutVgprsLeavesTheWord) {
    VgprMsbTracker tracker;
    tracker.step(kWordA, /*hasVgpr=*/true);
    EXPECT_FALSE(tracker.step(-1, /*hasVgpr=*/false).emit);
    EXPECT_FALSE(tracker.step(kWordA, /*hasVgpr=*/true).emit);
}

TEST(VgprMsbTrackerTest, BankZeroEverywhereIsAWordToo) {
    // Word 0 still has to be set at the start, and is the previous word of the
    // next switch, which Msb16 packs.
    VgprMsbTracker tracker;
    EXPECT_TRUE(tracker.step(0, /*hasVgpr=*/true).emit);
    EXPECT_FALSE(tracker.step(0, /*hasVgpr=*/true).emit);
    EXPECT_EQ(tracker.step(kWordA, /*hasVgpr=*/true).previousWord, 0);
}

TEST(VgprMsbTrackerTest, ALabelForgetsTheWordAndAsksForAnSNop) {
    VgprMsbTracker tracker;
    tracker.step(kWordA, /*hasVgpr=*/true);
    tracker.label();
    const VgprMsbStep again = tracker.step(kWordA, /*hasVgpr=*/true);
    EXPECT_TRUE(again.emit);
    EXPECT_TRUE(again.firstAfterLabel);
    EXPECT_FALSE(again.previousWord.has_value());
}

TEST(VgprMsbTrackerTest, AnInstructionAfterTheLabelDropsTheSNop) {
    VgprMsbTracker tracker;
    tracker.label();
    EXPECT_FALSE(tracker.step(-1, /*hasVgpr=*/false).emit);
    const VgprMsbStep set = tracker.step(kWordA, /*hasVgpr=*/true);
    EXPECT_TRUE(set.emit);
    EXPECT_FALSE(set.firstAfterLabel);
    EXPECT_FALSE(set.previousWord.has_value());
}

TEST(VgprMsbTrackerTest, AResetForgetsTheWordWithoutAnSNop) {
    VgprMsbTracker tracker;
    tracker.step(kWordA, /*hasVgpr=*/true);
    tracker.reset();
    const VgprMsbStep again = tracker.step(kWordA, /*hasVgpr=*/true);
    EXPECT_TRUE(again.emit);
    EXPECT_FALSE(again.firstAfterLabel);
    EXPECT_FALSE(again.previousWord.has_value());
}

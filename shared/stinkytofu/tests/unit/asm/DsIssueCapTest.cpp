// Copyright (C) 2025-2026 Advanced Micro Devices, Inc.

#include <gtest/gtest.h>

#include "transforms/asm/dag/DsIssueCap.hpp"

using namespace stinkytofu;
using Mode = PassFeatureConfig::DsIssueCapMode;

TEST(DsIssueCap, BothModesFillAtDepthAndFreeAfterSpan) {
    for (Mode mode : {Mode::Sliding, Mode::Periodic}) {
        DsIssueCap cap(mode, /*depth=*/3);
        for (int i = 0; i < 3; ++i) {
            EXPECT_FALSE(cap.full());
            cap.push(/*span=*/10);
        }
        EXPECT_TRUE(cap.full());
        EXPECT_EQ(cap.minResidual(), 10);
        cap.advance(10);
        EXPECT_FALSE(cap.full());
    }
}

TEST(DsIssueCap, TightBurstsAgreeAcrossModes) {
    // Back-to-back bursts of A: both modes release the whole burst X after
    // its first ds_load.
    for (Mode mode : {Mode::Sliding, Mode::Periodic}) {
        DsIssueCap cap(mode, /*depth=*/4);
        for (int i = 0; i < 4; ++i) cap.push(/*span=*/32);  // all at cycle 0
        ASSERT_TRUE(cap.full());
        EXPECT_EQ(cap.minResidual(), 32);
    }
}

TEST(DsIssueCap, SlidingFreesEachSlotAfterItsOwnIssue) {
    DsIssueCap cap(Mode::Sliding, /*depth=*/2);
    cap.push(/*span=*/10);  // t=0
    cap.advance(6);
    cap.push(/*span=*/10);  // t=6
    ASSERT_TRUE(cap.full());
    EXPECT_EQ(cap.minResidual(), 4);
    cap.advance(4);  // t=10: the first slot frees, one slot is open
    EXPECT_FALSE(cap.full());
    cap.push(10);  // t=10
    EXPECT_TRUE(cap.full());
    EXPECT_EQ(cap.minResidual(), 6);  // next free at the t=6 entry, not a whole period
}

TEST(DsIssueCap, PeriodicFreesAllSlotsAtTheEndOfThePeriod) {
    DsIssueCap cap(Mode::Periodic, /*depth=*/2);
    cap.push(/*span=*/10);  // t=0 opens the period
    cap.advance(6);
    cap.push(10);  // t=6 straggler, same period
    ASSERT_TRUE(cap.full());
    EXPECT_EQ(cap.minResidual(), 4);  // period ends at t=10, not t=16
    cap.advance(4);
    EXPECT_FALSE(cap.full());
    cap.push(10);  // t=10 opens a new period
    EXPECT_FALSE(cap.full());
    cap.push(10);
    EXPECT_TRUE(cap.full());
    EXPECT_EQ(cap.minResidual(), 10);
}

TEST(DsIssueCap, PeriodicBurstsStartExactlyOnePeriodApart) {
    DsIssueCap cap(Mode::Periodic, /*depth=*/12);
    int start1 = 0, start2 = 0, now = 0;
    for (int burst = 0; burst < 2; ++burst) {
        for (int i = 0; i < 12; ++i) {
            if (cap.full()) {
                const int wait = cap.minResidual();
                cap.advance(wait);
                now += wait;
            }
            if (i == 0) (burst == 0 ? start1 : start2) = now;
            cap.push(/*span=*/32);
            cap.advance(1);
            ++now;
        }
    }
    EXPECT_EQ(start2 - start1, 32);
}

TEST(DsIssueCap, DefaultModeIsSliding) {
    EXPECT_EQ(DsIssueCap().mode(), Mode::Sliding);
}

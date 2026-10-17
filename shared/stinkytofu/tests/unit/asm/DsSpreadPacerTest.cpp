// Copyright (C) 2025-2026 Advanced Micro Devices, Inc.

#include <gtest/gtest.h>

#include <climits>

#include "transforms/asm/dag/DsSpreadPacer.hpp"

using namespace stinkytofu;

namespace {
std::vector<DsSpreadPacer::Load> loads(int n, int deadline = INT_MAX) {
    std::vector<DsSpreadPacer::Load> v;
    for (int i = 0; i < n; ++i) v.push_back({static_cast<unsigned>(i), deadline});
    return v;
}

// Issues released loads from `next` on until the batch is full; returns how many.
int issueBatch(DsSpreadPacer& p, int wmmaIssued, unsigned& next, unsigned total) {
    int n = 0;
    while (next < total && p.released(next, wmmaIssued)) {
        p.noteIssue(next, wmmaIssued);
        ++next;
        ++n;
    }
    return n;
}
}  // namespace

TEST(DsSpreadPacer, SparseRegionTakesItsEvenShare) {
    // 8 loads over 32 WMMAs in batches of 4: 1 per batch, well under what a batch can hold.
    DsSpreadPacer p;
    p.plan(loads(8), /*wmmaTotal=*/32, /*batch=*/4, /*perBatch=*/5, /*catchUp=*/2, /*drain=*/0);
    unsigned next = 0;
    for (int w = 0; w < 32; w += 4) EXPECT_EQ(issueBatch(p, w, next, 8), 1) << "w=" << w;
}

TEST(DsSpreadPacer, DenseRegionIsCappedAtWhatABatchHoldsPlusCatchUp) {
    // 60 loads over 32 WMMAs would need 8 per batch: capped at perBatch + catchUp = 7.
    DsSpreadPacer p;
    p.plan(loads(60), 32, 4, 5, 2, 0);
    unsigned next = 0;
    EXPECT_EQ(issueBatch(p, 0, next, 60), 7);
}

TEST(DsSpreadPacer, ABatchThatMissedItsLoadsIsFollowedByAFasterOne) {
    // 16 loads over 32 WMMAs: 2 per batch. Nothing is ready in the first 4 batches (a
    // dependency); the share then grows to catch up, within perBatch + catchUp.
    DsSpreadPacer p;
    p.plan(loads(16), 32, 4, 3, 2, 0);
    unsigned next = 0;
    EXPECT_EQ(issueBatch(p, 16, next, 16), 4);  // 16 left over 4 batches
    EXPECT_EQ(issueBatch(p, 20, next, 16), 4);
}

TEST(DsSpreadPacer, TheDrainMovesTheRegionEndEarlier) {
    // 12 loads, 32 WMMAs, 8 of them kept free at the end: 24 usable WMMAs = 6 batches, 2 each.
    DsSpreadPacer p;
    p.plan(loads(12), 32, 4, 5, 2, /*drain=*/8);
    unsigned next = 0;
    EXPECT_EQ(issueBatch(p, 0, next, 12), 2);
}

TEST(DsSpreadPacer, ALoadWithAnEarlierDeadlineRaisesTheShare) {
    // 4 of 10 loads are due by WMMA 8 (2 batches): 2 per batch from the start, although the
    // region as a whole would only ask for 1.
    std::vector<DsSpreadPacer::Load> v;
    for (int i = 0; i < 4; ++i) v.push_back({static_cast<unsigned>(i), 8});
    for (int i = 4; i < 10; ++i) v.push_back({static_cast<unsigned>(i), INT_MAX});
    DsSpreadPacer p;
    p.plan(v, 40, 4, 5, 2, 0);
    unsigned next = 0;
    EXPECT_EQ(issueBatch(p, 0, next, 10), 2);
}

TEST(DsSpreadPacer, ALoadPastItsDeadlineIsAlwaysReleased) {
    DsSpreadPacer p;
    p.plan(loads(8, /*deadline=*/4), 32, 4, 1, 0, 0);
    unsigned next = 0;
    EXPECT_EQ(issueBatch(p, 0, next, 8), 1);
    EXPECT_FALSE(p.released(next, 0));
    EXPECT_TRUE(p.released(next, 4));  // the deadline: no allowance applies
}

TEST(DsSpreadPacer, InactiveWithoutLoadsOrWmmas) {
    DsSpreadPacer p;
    p.plan({}, 32, 4, 5, 2, 0);
    EXPECT_FALSE(p.active());
    EXPECT_TRUE(p.released(0, 0));
}

TEST(DsSpreadPacer, WindowsForIsTheEvenSpreadRoundedUp) {
    EXPECT_EQ(DsSpreadPacer::windowsFor(3, 128, 140), 3);
    EXPECT_EQ(DsSpreadPacer::windowsFor(0, 128, 140), 0);
    EXPECT_EQ(DsSpreadPacer::windowsFor(10, 100, 0), 0);
}

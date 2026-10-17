// Copyright (C) 2025-2026 Advanced Micro Devices, Inc.

#include <gtest/gtest.h>

#include <numeric>

#include "transforms/asm/dag/DsSpreadPacer.hpp"

using namespace stinkytofu;

namespace {
std::vector<unsigned> loads(int n) {
    std::vector<unsigned> v(n);
    std::iota(v.begin(), v.end(), 0u);
    return v;
}

// Issues loads while the batch has room; returns how many.
int issueBatch(DsSpreadPacer& p, int wmmaIssued, int cap = 1000) {
    int n = 0;
    while (n < cap && p.released(wmmaIssued)) {
        p.noteIssue(wmmaIssued);
        ++n;
    }
    return n;
}
}  // namespace

TEST(DsSpreadPacer, SparseRegionTakesItsEvenShare) {
    // 8 loads over 32 WMMAs in batches of 4: 1 per batch.
    DsSpreadPacer p;
    p.plan(loads(8), /*wmmaTotal=*/32, /*batch=*/4, /*catchUp=*/2, /*drain=*/0);
    for (int w = 0; w < 32; w += 4) EXPECT_EQ(issueBatch(p, w, 8), 1) << "w=" << w;
}

TEST(DsSpreadPacer, ADenseRegionGetsAShareAboveWhatTheLdsRateFits) {
    // 60 loads over 32 WMMAs: 8 batches, ceil(60 / 8) = 8 per batch.
    DsSpreadPacer p;
    p.plan(loads(60), 32, 4, 2, 0);
    EXPECT_EQ(issueBatch(p, 0), 8);
}

TEST(DsSpreadPacer, TheShareDoesNotGrowWhileOnTheEvenLine) {
    // 128 loads, 109 WMMAs, 17 kept free: 23 batches, 6 per batch every batch.
    DsSpreadPacer p;
    p.plan(loads(128), 109, 4, 2, 17);
    for (int w = 0; w < 40; w += 4) EXPECT_EQ(issueBatch(p, w), 6) << "w=" << w;
}

TEST(DsSpreadPacer, ABatchThatBeganBehindTakesTheCatchUpStep) {
    // 16 loads over 32 WMMAs: 2 per batch. Nothing was ready in the first 4 batches (a
    // dependency): every batch from the fifth on begins behind the even line and takes 2 + 2.
    DsSpreadPacer p;
    p.plan(loads(16), 32, 4, 2, 0);
    for (int w = 16; w < 32; w += 4) EXPECT_EQ(issueBatch(p, w, 4), 4) << "w=" << w;
}

TEST(DsSpreadPacer, TheDrainMovesTheRegionEndEarlier) {
    // 12 loads, 32 WMMAs, 8 of them kept free at the end: 24 usable WMMAs = 6 batches, 2 each.
    DsSpreadPacer p;
    p.plan(loads(12), 32, 4, 2, /*drain=*/8);
    EXPECT_EQ(issueBatch(p, 0), 2);
}

TEST(DsSpreadPacer, NothingIsHeldOnceTheRegionsWmmasHaveIssued) {
    DsSpreadPacer p;
    p.plan(loads(8), 32, 4, 0, 0);
    EXPECT_EQ(issueBatch(p, 0), 1);
    EXPECT_TRUE(p.released(32));
}

TEST(DsSpreadPacer, InactiveWithoutLoadsOrWmmas) {
    DsSpreadPacer p;
    p.plan({}, 32, 4, 2, 0);
    EXPECT_FALSE(p.active());
    EXPECT_TRUE(p.released(0));
}

TEST(DsSpreadPacer, WindowsForIsTheEvenSpreadRoundedUp) {
    EXPECT_EQ(DsSpreadPacer::windowsFor(3, 128, 140), 3);
    EXPECT_EQ(DsSpreadPacer::windowsFor(0, 128, 140), 0);
    EXPECT_EQ(DsSpreadPacer::windowsFor(10, 100, 0), 0);
}

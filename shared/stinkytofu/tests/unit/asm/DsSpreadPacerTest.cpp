// Copyright (C) 2025-2026 Advanced Micro Devices, Inc.

#include <gtest/gtest.h>

#include <climits>
#include <deque>

#include "transforms/asm/dag/DsSpreadPacer.hpp"

using namespace stinkytofu;

namespace {
using Loads = std::vector<DsSpreadPacer::Load>;

// n loads with ids first.., all in the stage [start, end).
void add(Loads& v, int n, unsigned first, int start = 0, int end = INT_MAX) {
    for (int i = 0; i < n; ++i) v.push_back({first + static_cast<unsigned>(i), start, end});
}

// Issues the pending loads while their stage has room in the batch at `wmmaIssued`; returns how
// many went.
int issueBatch(DsSpreadPacer& p, int wmmaIssued, std::deque<unsigned>& pending, int cap = 1000) {
    int n = 0;
    while (n < cap && !pending.empty() && p.released(pending.front(), wmmaIssued)) {
        p.noteIssue(pending.front(), wmmaIssued);
        pending.pop_front();
        ++n;
    }
    return n;
}

std::deque<unsigned> ids(const Loads& v) {
    std::deque<unsigned> d;
    for (const auto& l : v) d.push_back(l.nodeId);
    return d;
}
}  // namespace

TEST(DsSpreadPacer, SparseStageTakesItsEvenShare) {
    // 8 loads over 32 WMMAs in batches of 4: 1 per batch.
    Loads v;
    add(v, 8, 0);
    DsSpreadPacer p;
    p.plan(v, /*wmmaTotal=*/32, /*batch=*/4, /*catchUp=*/2, /*drain=*/0);
    auto pending = ids(v);
    for (int w = 0; w < 32; w += 4) EXPECT_EQ(issueBatch(p, w, pending), 1) << "w=" << w;
}

TEST(DsSpreadPacer, ADenseStageGetsAShareAboveWhatTheLdsRateFits) {
    // 60 loads over 32 WMMAs: 8 batches, ceil(60 / 8) = 8 per batch.
    Loads v;
    add(v, 60, 0);
    DsSpreadPacer p;
    p.plan(v, 32, 4, 2, 0);
    auto pending = ids(v);
    EXPECT_EQ(issueBatch(p, 0, pending), 8);
}

TEST(DsSpreadPacer, TheShareDoesNotGrowWhileOnTheEvenLine) {
    // 128 loads, 109 WMMAs, 17 kept free: 23 batches, 6 per batch every batch.
    Loads v;
    add(v, 128, 0);
    DsSpreadPacer p;
    p.plan(v, 109, 4, 2, 17);
    auto pending = ids(v);
    for (int w = 0; w < 40; w += 4) EXPECT_EQ(issueBatch(p, w, pending), 6) << "w=" << w;
}

TEST(DsSpreadPacer, ABatchThatBeganBehindTakesTheCatchUpStep) {
    // 16 loads over 32 WMMAs: 2 per batch. Nothing was ready in the first 4 batches (a
    // dependency): every batch from the fifth on begins behind the even line and takes 2 + 2.
    Loads v;
    add(v, 16, 0);
    DsSpreadPacer p;
    p.plan(v, 32, 4, 2, 0);
    auto pending = ids(v);
    for (int w = 16; w < 32; w += 4) EXPECT_EQ(issueBatch(p, w, pending, 4), 4) << "w=" << w;
}

TEST(DsSpreadPacer, EachStageHasItsOwnWindowAndShare) {
    // Stage A: 8 loads before the barrier at WMMA 40 (10 batches, 1 each). Stage B: 12 loads
    // behind it, in the 24 WMMAs left (6 batches, 2 each), from batch 10 on.
    Loads v;
    add(v, 8, 0, 0, 40);
    add(v, 12, 100, 40, INT_MAX);
    DsSpreadPacer p;
    p.plan(v, 64, 4, 2, 0);
    std::deque<unsigned> a, b;
    for (const auto& l : v) (l.nodeId < 100 ? a : b).push_back(l.nodeId);
    EXPECT_EQ(issueBatch(p, 0, a), 1);
    EXPECT_EQ(issueBatch(p, 40, b), 2);  // not behind: its line starts at its own first batch
    EXPECT_EQ(issueBatch(p, 44, b), 2);
}

TEST(DsSpreadPacer, TheDrainEndsAStageEarlier) {
    // 12 loads, 32 WMMAs, 8 of them kept free before the region end: 24 WMMAs = 6 batches.
    Loads v;
    add(v, 12, 0);
    DsSpreadPacer p;
    p.plan(v, 32, 4, 2, /*drain=*/8);
    auto pending = ids(v);
    EXPECT_EQ(issueBatch(p, 0, pending), 2);
}

TEST(DsSpreadPacer, NothingIsHeldOnceTheRegionsWmmasHaveIssued) {
    Loads v;
    add(v, 8, 0);
    DsSpreadPacer p;
    p.plan(v, 32, 4, 0, 0);
    auto pending = ids(v);
    EXPECT_EQ(issueBatch(p, 0, pending), 1);
    EXPECT_TRUE(p.released(pending.front(), 32));
}

TEST(DsSpreadPacer, InactiveWithoutLoadsOrWmmas) {
    DsSpreadPacer p;
    p.plan({}, 32, 4, 2, 0);
    EXPECT_FALSE(p.active());
    EXPECT_TRUE(p.released(0, 0));
}

TEST(DsSpreadPacer, WindowsForIsTheEvenSpreadRoundedUp) {
    EXPECT_EQ(DsSpreadPacer::windowsFor(3, 128, 140), 3);
    EXPECT_EQ(DsSpreadPacer::windowsFor(0, 128, 140), 0);
    EXPECT_EQ(DsSpreadPacer::windowsFor(10, 100, 0), 0);
}

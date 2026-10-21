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

TEST(DsSpreadPacer, TheQuotaFollowsTheEvenLine) {
    // 128 loads feeding a barrier at WMMA 109, 17 kept free: 23 batches on the even line, 5 or 6
    // per batch (128 / 23 = 5.57), 56 placed after the first 10.
    Loads v;
    add(v, 128, 0, 0, 109);
    DsSpreadPacer p;
    p.plan(v, 109, 4, 2, 17);
    auto pending = ids(v);
    int placed = 0;
    for (int w = 0; w < 40; w += 4) {
        const int n = issueBatch(p, w, pending);
        EXPECT_TRUE(n == 5 || n == 6) << "w=" << w << " n=" << n;
        placed += n;
    }
    EXPECT_EQ(placed, 56);
}

TEST(DsSpreadPacer, ABatchThatBeganBehindTakesExactlyTheShareAndTheCatchUpStep) {
    // 16 loads over 32 WMMAs: 2 per batch. Nothing was ready in the first 4 batches (a
    // dependency): every batch from the fifth on begins behind the even line and takes 2 + 2.
    Loads v;
    add(v, 16, 0);
    DsSpreadPacer p;
    p.plan(v, 32, 4, 2, 0);
    auto pending = ids(v);
    EXPECT_EQ(issueBatch(p, 16, pending), 4);
    EXPECT_EQ(issueBatch(p, 20, pending), 4);
    EXPECT_EQ(issueBatch(p, 24, pending), 4);
    EXPECT_EQ(issueBatch(p, 28, pending), 4);
    EXPECT_TRUE(pending.empty());
}

TEST(DsSpreadPacer, CatchingUpStopsOnceTheStageIsOnItsLine) {
    // 8 loads, 2 per batch over 16 WMMAs; the first batch is skipped, the second takes 4 and
    // is back on the line (4 of 4 expected), so the third takes the plain share again.
    Loads v;
    add(v, 8, 0);
    DsSpreadPacer p;
    p.plan(v, 16, 4, 2, 0);
    auto pending = ids(v);
    EXPECT_EQ(issueBatch(p, 4, pending), 4);
    EXPECT_EQ(issueBatch(p, 8, pending), 2);
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

TEST(DsSpreadPacer, StagesOpenAtTheSameTimeEachTakeTheirOwnShare) {
    // Documented behaviour: stages are paced independently, so two stages open in one batch add
    // their shares (1 + 1 here).
    Loads v;
    add(v, 8, 0, 0, 40);
    add(v, 8, 100, 0, 40 + 1);
    DsSpreadPacer p;
    p.plan(v, 64, 4, 0, 0);
    std::deque<unsigned> a, b;
    for (const auto& l : v) (l.nodeId < 100 ? a : b).push_back(l.nodeId);
    EXPECT_EQ(issueBatch(p, 0, a), 1);
    EXPECT_EQ(issueBatch(p, 0, b), 1);
}

TEST(DsSpreadPacer, TheDrainEndsAStageEarlier) {
    // 12 loads feeding a barrier at WMMA 32, 8 of them kept free in front of it: 24 WMMAs = 6
    // batches, so 2 per batch in every one of them.
    Loads v;
    add(v, 12, 0, 0, 32);
    DsSpreadPacer p;
    p.plan(v, 32, 4, 0, /*drain=*/8);
    auto pending = ids(v);
    for (int w = 0; w < 24; w += 4) EXPECT_EQ(issueBatch(p, w, pending), 2) << "w=" << w;
    EXPECT_TRUE(pending.empty());
}

TEST(DsSpreadPacer, AStageEndsWithItsLastBatch) {
    // 63 loads over 12 batches (48 WMMAs): 5 5 5 6 repeating, all placed by the last batch.
    Loads v;
    add(v, 63, 0, 0, 48 + 17);
    DsSpreadPacer p;
    p.plan(v, 128, 4, 0, 17);
    auto pending = ids(v);
    int total = 0;
    for (int w = 0; w < 48; w += 4) {
        const int n = issueBatch(p, w, pending);
        EXPECT_TRUE(n == 5 || n == 6) << "w=" << w << " n=" << n;
        total += n;
    }
    EXPECT_EQ(total, 63);
    EXPECT_TRUE(pending.empty());
}

TEST(DsSpreadPacer, TheRegionEndNeedsNoDrain) {
    // 16 loads to the region end (the loop returns to its start), 32 WMMAs, 8 of them the drain.
    // Without a drain at the region end the stage has all 8 batches: 2 loads in each, the last
    // one in the batch at WMMA 28. With the drain kept it would have 6 batches (3 loads each) and
    // be done by WMMA 20.
    Loads v;
    add(v, 16, 0);
    DsSpreadPacer p;
    p.plan(v, 32, 4, 0, /*drain=*/8);
    auto pending = ids(v);
    for (int w = 0; w < 32; w += 4) EXPECT_EQ(issueBatch(p, w, pending), 2) << "w=" << w;
    EXPECT_TRUE(pending.empty());
}

TEST(DsSpreadPacer, AStageEndsWhereItsLastLoadCanFirstIssue) {
    // 12 loads feeding a barrier at WMMA 20 with 8 kept free: planned end 12. The graph says the
    // last load cannot issue before 24, so the stage ends there: 6 batches of 2 loads.
    Loads v;
    add(v, 12, 0, 0, 20);
    v.back().earliest = 24;
    DsSpreadPacer p;
    p.plan(v, 64, 4, 0, 8);
    auto pending = ids(v);
    for (int w = 0; w < 24; w += 4) EXPECT_EQ(issueBatch(p, w, pending), 2) << "w=" << w;
    EXPECT_TRUE(pending.empty());
}

TEST(DsSpreadPacer, AStageNotLaterThanItsPlanKeepsItsPlannedEnd) {
    // The last load can issue by 8, before the planned end 12: the stage keeps the planned end.
    Loads v;
    add(v, 12, 0, 0, 20);
    v.back().earliest = 8;
    DsSpreadPacer p;
    p.plan(v, 64, 4, 0, 8);
    auto pending = ids(v);
    for (int w = 0; w < 12; w += 4) EXPECT_EQ(issueBatch(p, w, pending), 4) << "w=" << w;
    EXPECT_TRUE(pending.empty());
}

TEST(DsSpreadPacer, ABatchBehindItsLineTakesWhatTheStageNeedsUpToTheHardware) {
    // 24 loads over 16 WMMAs: 6 per batch. Nothing issued in the first two batches (a dependency),
    // so the third begins behind its line. The stage needs 12 in the 2 batches left; the hardware
    // takes 8 per batch: 8. Without the bound it is the quota plus the catch-up step: 6.
    for (int cap : {0, 8}) {
        Loads v;
        add(v, 24, 0, 0, INT_MAX);
        DsSpreadPacer p;
        p.plan(v, 16, 4, 0, 0, cap);
        auto pending = ids(v);
        EXPECT_EQ(issueBatch(p, 8, pending), cap == 0 ? 6 : 8) << "cap=" << cap;
    }
}

TEST(DsSpreadPacer, AStageTheDrainLeavesNoRoomForIsReleasedInOneBatch) {
    // The stage starts at WMMA 115 of 128 and feeds a barrier at 128 with 17 kept free: its end
    // (111) falls before its start, so it has one batch and all 35 loads are due in it.
    Loads v;
    add(v, 35, 0, 115, 128);
    DsSpreadPacer p;
    p.plan(v, 128, 4, 2, 17);
    auto pending = ids(v);
    EXPECT_EQ(issueBatch(p, 116, pending), 35);
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

TEST(DsSpreadPacer, ALoadOutsideThePlanIsAlwaysReleased) {
    Loads v;
    add(v, 4, 0);
    DsSpreadPacer p;
    p.plan(v, 16, 4, 2, 0);
    EXPECT_TRUE(p.released(999, 0));
    p.noteIssue(999, 0);  // harmless
    EXPECT_TRUE(p.released(999, 0));
}

TEST(DsSpreadPacer, NonPositiveBatchAndNegativeCatchUpAreClamped) {
    Loads v;
    add(v, 4, 0);
    DsSpreadPacer p;
    p.plan(v, 4, /*batch=*/0, /*catchUp=*/-3, 0);  // batch 1, catch-up 0: 1 load per WMMA
    auto pending = ids(v);
    for (int w = 0; w < 4; ++w) EXPECT_EQ(issueBatch(p, w, pending), 1) << "w=" << w;
}

TEST(DsSpreadPacer, ARegionWithoutWmmasIsInactive) {
    Loads v;
    add(v, 4, 0);
    DsSpreadPacer p;
    p.plan(v, 0, 4, 2, 0);
    EXPECT_FALSE(p.active());
    EXPECT_TRUE(p.released(0, 0));
}

TEST(DsSpreadPacer, PlanningAgainStartsFresh) {
    Loads v;
    add(v, 8, 0);
    DsSpreadPacer p;
    p.plan(v, 32, 4, 2, 0);
    auto pending = ids(v);
    EXPECT_EQ(issueBatch(p, 0, pending), 1);
    p.plan(v, 32, 4, 2, 0);
    pending = ids(v);
    EXPECT_EQ(issueBatch(p, 0, pending), 1);  // the previous region's counts are gone
}

TEST(DsSpreadPacer, InactiveWithoutLoads) {
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

TEST(DsSpreadPacer, AStageOpeningOffABatchBoundaryCountsBatchesFromItsStart) {
    // start 3, end 11, batch 4, 6 loads: two full batches (WMMAs 3..6 and 7..10), 3 loads each.
    Loads v;
    add(v, 6, 0, /*start=*/3, /*end=*/11);
    DsSpreadPacer p;
    p.plan(v, /*wmmaTotal=*/16, /*batch=*/4, /*catchUp=*/0, /*drain=*/0);
    auto pending = ids(v);
    std::vector<int> per;
    for (int w = 3; w < 11; ++w) per.push_back(issueBatch(p, w, pending));
    EXPECT_EQ(per, (std::vector<int>{3, 0, 0, 0, 3, 0, 0, 0}));
}

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

// The drain every load's barrier keeps free (the scheduler gives each load its barrier's).
void setDrain(Loads& v, int drain) {
    for (auto& l : v) l.drain = drain;
}

void setOpener(Loads& v, unsigned opener) {
    for (auto& l : v) l.opener = opener;
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
    p.plan(v, /*wmmaTotal=*/32, /*batch=*/4, /*catchUp=*/2);
    auto pending = ids(v);
    for (int w = 0; w < 32; w += 4) EXPECT_EQ(issueBatch(p, w, pending), 1) << "w=" << w;
}

TEST(DsSpreadPacer, ADenseStageGetsAShareAboveWhatTheLdsRateFits) {
    // 60 loads over 32 WMMAs: 8 batches, ceil(60 / 8) = 8 per batch.
    Loads v;
    add(v, 60, 0);
    DsSpreadPacer p;
    p.plan(v, 32, 4, 2);
    auto pending = ids(v);
    EXPECT_EQ(issueBatch(p, 0, pending), 8);
}

TEST(DsSpreadPacer, TheQuotaFollowsTheEvenLine) {
    // 128 loads feeding a barrier at WMMA 109, 17 kept free: 23 batches on the even line, 5 or 6
    // per batch (128 / 23 = 5.57), 56 placed after the first 10.
    Loads v;
    add(v, 128, 0, 0, 109);
    DsSpreadPacer p;
    setDrain(v, 17);
    p.plan(v, 109, 4, 2);
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
    p.plan(v, 32, 4, 2);
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
    p.plan(v, 16, 4, 2);
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
    p.plan(v, 64, 4, 2);
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
    p.plan(v, 64, 4, 0);
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
    setDrain(v, 8);
    p.plan(v, 32, 4, 0);
    auto pending = ids(v);
    for (int w = 0; w < 24; w += 4) EXPECT_EQ(issueBatch(p, w, pending), 2) << "w=" << w;
    EXPECT_TRUE(pending.empty());
}

TEST(DsSpreadPacer, AStageEndsWithItsLastBatch) {
    // 63 loads over 12 batches (48 WMMAs): 5 5 5 6 repeating, all placed by the last batch.
    Loads v;
    add(v, 63, 0, 0, 48 + 17);
    DsSpreadPacer p;
    setDrain(v, 17);
    p.plan(v, 128, 4, 0);
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

TEST(DsSpreadPacer, ALoadUsesTheDrainItsBarrierWasPlannedWith) {
    // 12 loads feeding a barrier at WMMA 20. A barrier planned with a drain of 4 (N = last ds + 4)
    // ends the stage at 16: 4 batches of 3. A drain of 8 ends it at 12: 3 batches of 4.
    for (int drain : {8, 4}) {
        Loads v;
        add(v, 12, 0, 0, 20);
        setDrain(v, drain);
        DsSpreadPacer p;
        p.plan(v, 64, 4, 0);
        auto pending = ids(v);
        const int perBatch = drain == 4 ? 3 : 4;
        for (int w = 0; w < 12; w += 4) EXPECT_EQ(issueBatch(p, w, pending), perBatch) << "w=" << w;
    }
}

TEST(DsSpreadPacer, TheRegionEndNeedsNoDrain) {
    // 16 loads to the region end (the loop returns to its start), 32 WMMAs, 8 of them the drain.
    // Without a drain at the region end the stage has all 8 batches: 2 loads in each, the last
    // one in the batch at WMMA 28. With the drain kept it would have 6 batches (3 loads each) and
    // be done by WMMA 20.
    Loads v;
    add(v, 16, 0);
    DsSpreadPacer p;
    setDrain(v, 8);
    p.plan(v, 32, 4, 0);
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
    setDrain(v, 8);
    p.plan(v, 64, 4, 0);
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
    setDrain(v, 8);
    p.plan(v, 64, 4, 0);
    auto pending = ids(v);
    for (int w = 0; w < 12; w += 4) EXPECT_EQ(issueBatch(p, w, pending), 4) << "w=" << w;
    EXPECT_TRUE(pending.empty());
}

TEST(DsSpreadPacer, AStageEndsWhereTheLdsRateFitsItsLoadsWhenThatIsBeforeItsPlannedEnd) {
    // 64 loads from WMMA 0 to a barrier at 71 with 17 kept free: planned end 54. At 6 cycles per
    // load and 8 cycles per WMMA the loads fit by 48, so the stage ends there: 12 batches, 6 loads
    // in the first. Without the rate the stage ends at 54: 14 batches, 5 in the first.
    for (bool rate : {false, true}) {
        Loads v;
        add(v, 64, 0, 0, 71);
        v.back().earliest = 40;
        DsSpreadPacer p;
        setDrain(v, 17);
        p.plan(v, 128, 4, 0, 0, rate ? 6 : 0, rate ? 8 : 0);
        auto pending = ids(v);
        EXPECT_EQ(issueBatch(p, 0, pending), rate ? 6 : 5) << "rate=" << rate;
    }
}

TEST(DsSpreadPacer, ARateFloorAtOrAfterThePlannedEndChangesNothing) {
    // 64 loads from WMMA 80 to the region end (128): the loads fit by 80 + 48 = 128, the planned
    // end, so the stage keeps its 12 batches.
    for (bool rate : {false, true}) {
        Loads v;
        add(v, 64, 0, 80);
        DsSpreadPacer p;
        setDrain(v, 17);
        p.plan(v, 128, 4, 0, 0, rate ? 6 : 0, rate ? 8 : 0);
        auto pending = ids(v);
        EXPECT_EQ(issueBatch(p, 80, pending), 6) << "rate=" << rate;
    }
}

TEST(DsSpreadPacer, ABatchBehindItsLineTakesWhatTheStageNeedsUpToTheHardware) {
    // 24 loads over 16 WMMAs: 6 per batch. Nothing issued in the first two batches (a dependency),
    // so the third begins behind its line. The stage needs 12 in the 2 batches left; the hardware
    // takes 8 per batch: 8. Without the bound it is the quota plus the catch-up step: 6.
    for (int cap : {0, 8}) {
        Loads v;
        add(v, 24, 0, 0, INT_MAX);
        DsSpreadPacer p;
        p.plan(v, 16, 4, 0, cap);
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
    setDrain(v, 17);
    p.plan(v, 128, 4, 2);
    auto pending = ids(v);
    EXPECT_EQ(issueBatch(p, 116, pending), 35);
}

TEST(DsSpreadPacer, NothingIsHeldOnceTheRegionsWmmasHaveIssued) {
    Loads v;
    add(v, 8, 0);
    DsSpreadPacer p;
    p.plan(v, 32, 4, 0);
    auto pending = ids(v);
    EXPECT_EQ(issueBatch(p, 0, pending), 1);
    EXPECT_TRUE(p.released(pending.front(), 32));
}

TEST(DsSpreadPacer, ALoadOutsideThePlanIsAlwaysReleased) {
    Loads v;
    add(v, 4, 0);
    DsSpreadPacer p;
    p.plan(v, 16, 4, 2);
    EXPECT_TRUE(p.released(999, 0));
    p.noteIssue(999, 0);  // harmless
    EXPECT_TRUE(p.released(999, 0));
}

TEST(DsSpreadPacer, NonPositiveBatchAndNegativeCatchUpAreClamped) {
    Loads v;
    add(v, 4, 0);
    DsSpreadPacer p;
    p.plan(v, 4, /*batch=*/0, /*catchUp=*/-3);  // batch 1, catch-up 0: 1 load per WMMA
    auto pending = ids(v);
    for (int w = 0; w < 4; ++w) EXPECT_EQ(issueBatch(p, w, pending), 1) << "w=" << w;
}

TEST(DsSpreadPacer, ARegionWithoutWmmasIsInactive) {
    Loads v;
    add(v, 4, 0);
    DsSpreadPacer p;
    p.plan(v, 0, 4, 2);
    EXPECT_FALSE(p.active());
    EXPECT_TRUE(p.released(0, 0));
}

TEST(DsSpreadPacer, PlanningAgainStartsFresh) {
    Loads v;
    add(v, 8, 0);
    DsSpreadPacer p;
    p.plan(v, 32, 4, 2);
    auto pending = ids(v);
    EXPECT_EQ(issueBatch(p, 0, pending), 1);
    p.plan(v, 32, 4, 2);
    pending = ids(v);
    EXPECT_EQ(issueBatch(p, 0, pending), 1);  // the previous region's counts are gone
}

TEST(DsSpreadPacer, InactiveWithoutLoads) {
    DsSpreadPacer p;
    p.plan({}, 32, 4, 2);
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
    p.plan(v, /*wmmaTotal=*/16, /*batch=*/4, /*catchUp=*/0);
    auto pending = ids(v);
    std::vector<int> per;
    for (int w = 3; w < 11; ++w) per.push_back(issueBatch(p, w, pending));
    EXPECT_EQ(per, (std::vector<int>{3, 0, 0, 0, 3, 0, 0, 0}));
}

TEST(DsSpreadPacer, AStageTheLdsRateCanNotFinishByItsEndIsReported) {
    // 60 loads from WMMA 0 to a barrier at 60 with 17 kept free: end 43. At 6 cycles per load and 8
    // cycles per WMMA they need 45; at 5 they need 38; with no rate nothing is checked.
    for (int cycles : {0, 5, 6}) {
        Loads v;
        add(v, 60, 0, 0, 60);
        DsSpreadPacer p;
        setDrain(v, 17);
        p.plan(v, 128, 4, 0, 0, cycles, cycles > 0 ? 8 : 0);
        if (cycles == 6) {
            ASSERT_EQ(p.rateLimited().size(), 1u);
            EXPECT_EQ(p.rateLimited()[0].start, 0);
            EXPECT_EQ(p.rateLimited()[0].end, 43);
            EXPECT_EQ(p.rateLimited()[0].total, 60);
            EXPECT_EQ(p.rateLimited()[0].needed, 45);
        } else {
            EXPECT_TRUE(p.rateLimited().empty()) << "cycles=" << cycles;
        }
    }
}

// 16 loads planned over [8, 40) in batches of 4: 8 batches, 2 per batch. Issuing at WMMA 16
// without moving the start is two batches behind the even line, so the batch takes its 2 plus
// the catch-up of 2. The opening barrier issuing at 16 compresses the stage to [16, 40): 6
// batches, 3 then 3, and the stage is not behind.
TEST(DsSpreadPacer, ALateOpenerCompressesTheStageAndDropsTheFalseCatchUp) {
    Loads v;
    add(v, 16, 0, 8, 40);
    setOpener(v, 7);
    DsSpreadPacer late;
    late.plan(v, 64, 4, 2);
    auto caughtUp = ids(v);
    EXPECT_EQ(issueBatch(late, 16, caughtUp), 4);

    DsSpreadPacer moved;
    moved.plan(v, 64, 4, 2);
    moved.noteOpen(7, 16);
    auto pending = ids(v);
    EXPECT_EQ(moved.behind(pending.front(), 16), 0);
    EXPECT_EQ(issueBatch(moved, 16, pending), 3);
    EXPECT_EQ(issueBatch(moved, 20, pending), 3);
}

TEST(DsSpreadPacer, AnOnTimeOpenerKeepsThePlannedBatches) {
    Loads v;
    add(v, 16, 0, 8, 40);
    setOpener(v, 7);
    DsSpreadPacer p;
    p.plan(v, 64, 4, 2);
    p.noteOpen(7, 8);
    auto pending = ids(v);
    EXPECT_EQ(issueBatch(p, 8, pending), 2);
    EXPECT_EQ(issueBatch(p, 12, pending), 2);
}

TEST(DsSpreadPacer, ADifferentOpenerDoesNotMoveTheStage) {
    Loads v;
    add(v, 16, 0, 8, 40);
    setOpener(v, 7);
    DsSpreadPacer p;
    p.plan(v, 64, 4, 2);
    p.noteOpen(99, 16);
    auto pending = ids(v);
    EXPECT_EQ(issueBatch(p, 16, pending), 4);
}

TEST(DsSpreadPacer, MixedOpenersKeepThePlannedStart) {
    // The same planned window fed by two barriers stays one stage. Neither barrier may move it,
    // because that would also move the loads waiting behind the other one.
    Loads v;
    add(v, 8, 0, 8, 40);
    add(v, 8, 100, 8, 40);
    for (auto& l : v) l.opener = l.nodeId < 100 ? 7u : 9u;
    DsSpreadPacer p;
    p.plan(v, 64, 4, 2);
    p.noteOpen(7, 16);
    p.noteOpen(9, 16);
    auto pending = ids(v);
    EXPECT_EQ(issueBatch(p, 16, pending), 4);
}

TEST(DsSpreadPacer, AStageThatAlreadyIssuedKeepsItsStart) {
    // One load issued on the plan, then the opener is late. The batches stay as planned: at
    // WMMA 16 the stage is behind and takes the plain share (catch-up 0), not the 3 a compressed
    // stage would take.
    Loads v;
    add(v, 16, 0, 8, 40);
    setOpener(v, 7);
    DsSpreadPacer p;
    p.plan(v, 64, 4, 0);
    auto pending = ids(v);
    EXPECT_EQ(issueBatch(p, 8, pending, 1), 1);
    p.noteOpen(7, 16);
    EXPECT_EQ(issueBatch(p, 16, pending), 2);
}

TEST(DsSpreadPacer, AnOpenerPastThePlannedEndReleasesTheStage) {
    // The planned end is WMMA 20. The opener issues at 24, so the stage has no span left and
    // every remaining load is due in that one batch.
    Loads v;
    add(v, 8, 0, 8, 20);
    setOpener(v, 7);
    DsSpreadPacer p;
    p.plan(v, 64, 4, 0);
    p.noteOpen(7, 24);
    auto pending = ids(v);
    EXPECT_EQ(issueBatch(p, 24, pending), 8);
}

// 16 loads over [8, 40): the fill after a queue reset held every load until WMMA 16. The stage
// counts its batches from there, as for a late opener: 6 batches, 3 then 3, and not behind.
TEST(DsSpreadPacer, AStageHeldByTheQueueFillStartsWhereTheFillEnded) {
    Loads v;
    add(v, 16, 0, 8, 40);
    DsSpreadPacer p;
    p.plan(v, 64, 4, 2);
    p.noteHeldUntil(16);
    auto pending = ids(v);
    EXPECT_EQ(p.behind(pending.front(), 16), 0);
    EXPECT_EQ(issueBatch(p, 16, pending), 3);
    EXPECT_EQ(issueBatch(p, 20, pending), 3);
}

TEST(DsSpreadPacer, PlannedIssueWindowModelsAQueueFillWithoutMutatingThePlan) {
    Loads v;
    add(v, 16, 0, 8, 40);
    DsSpreadPacer p;
    p.plan(v, 64, 4, 2);

    // The hold restarts the unopened stage at 16: six batches, three loads in each of the first
    // four. Ordinal 3 therefore moves from its original second batch (12) to the new one (20).
    EXPECT_EQ(p.plannedIssueWindow(/*nodeId=*/3), 12);
    EXPECT_EQ(p.plannedIssueWindow(/*nodeId=*/3, /*heldUntil=*/16), 20);

    // The read-only estimate did not change the live pacer.
    auto pending = ids(v);
    EXPECT_EQ(issueBatch(p, 8, pending), 2);
}

TEST(DsSpreadPacer, PlannedIssueWindowStartsAnOpenStageFromTheHeldBatch) {
    Loads v;
    add(v, 8, 0, 0, 32);
    DsSpreadPacer p;
    p.plan(v, 32, 4, 0);
    auto pending = ids(v);
    EXPECT_EQ(issueBatch(p, 0, pending), 1);

    // The stage has already opened, so a later fill does not move its start. It does prevent
    // remaining loads from issuing before window 8; ordinal 1 is released in that batch.
    EXPECT_EQ(p.plannedIssueWindow(/*nodeId=*/1, /*heldUntil=*/8), 8);
}

TEST(DsSpreadPacer, TheQueueFillLeavesAStageThatIssuedOrHasNotOpened) {
    // [8, 40) issued a load on the plan and keeps its batches (behind at 16: its share of 2, no
    // catch-up). [24, 40) opens after the fill ended and keeps its 4 batches of 2; moved to 16 it
    // would have 6 and take 1 at 24.
    Loads v;
    add(v, 16, 0, 8, 40);
    add(v, 8, 100, 24, 40);
    DsSpreadPacer p;
    p.plan(v, 64, 4, 0);
    std::deque<unsigned> early{0}, rest, late;
    for (unsigned id = 1; id < 16; ++id) rest.push_back(id);
    for (unsigned id = 100; id < 108; ++id) late.push_back(id);
    EXPECT_EQ(issueBatch(p, 8, early), 1);
    p.noteHeldUntil(16);
    EXPECT_EQ(issueBatch(p, 16, rest), 2);
    EXPECT_EQ(issueBatch(p, 24, late), 2);
}

TEST(DsSpreadPacer, AStageIsDueAtItsEndWithLoadsLeft) {
    Loads v;
    add(v, 2, 0, 8, 20);
    DsSpreadPacer p;
    p.plan(v, 64, 4, 0);
    EXPECT_FALSE(p.due(0, 19));
    EXPECT_TRUE(p.due(0, 20));
    auto pending = ids(v);
    EXPECT_EQ(issueBatch(p, 20, pending), 2);
    EXPECT_FALSE(p.due(0, 20)) << "every load of the stage has issued";
}

TEST(DsSpreadPacer, TheRegionEndIsNeverDue) {
    // A stage closed by the region end has no barrier to make: nothing is held there.
    Loads v;
    add(v, 2, 0);
    DsSpreadPacer p;
    p.plan(v, 32, 4, 0);
    EXPECT_FALSE(p.due(0, 31));
    EXPECT_FALSE(p.due(0, 32));
}

TEST(DsSpreadPacer, TheRateLimitedWarningNamesTheStageTheRateAndTheShortfall) {
    DsSpreadPacer::RateLimited r{0, 43, 60, 45};
    EXPECT_EQ(DsSpreadPacer::rateLimitedMessage(r, 6),
              "stage [0, 43) has 60 ds_loads that need 45 WMMAs at 6 cycles per load but has 43: "
              "the pace is too sparse for the stage, its last ds_load will land late (lower "
              "DsReadCyclesPerLoad)");
}

TEST(DsSpreadPacer, ThePlanSummaryNamesEveryStageAndFlagsACollapsedOne) {
    // One stage spread over two batches, one whose window is shorter than a batch so every load
    // of it is released at once -- the case the summary exists to make visible.
    Loads v;
    add(v, 6, 0, /*start=*/3, /*end=*/11);
    add(v, 4, 6, /*start=*/12, /*end=*/13);
    DsSpreadPacer p;
    p.plan(v, /*wmmaTotal=*/16, /*batch=*/4, /*catchUp=*/0);
    EXPECT_EQ(p.planSummary(),
              "  stage [3, 11) loads=6 batches=2 lastDs=0 perBatch=3\n"
              "  stage [12, 13) loads=4 batches=1 lastDs=0 perBatch=4"
              "  <<< COLLAPSED (all at once)\n");
}

TEST(DsSpreadPacer, AnUnplannedPacerSummarisesNothing) {
    DsSpreadPacer p;
    EXPECT_EQ(p.planSummary(), "");
}

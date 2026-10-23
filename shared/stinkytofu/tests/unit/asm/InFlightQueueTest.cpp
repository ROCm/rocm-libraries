// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "transforms/asm/dag/InFlightQueue.hpp"

using namespace stinkytofu;

// ---------------------------------------------------------------------------
// Credit window semantics: depth is the number of credits, entry lifetime is how long one stays
// in flight, so full() means "depth credits issued within the last lifetime cycles". The ds_load
// credit queue of CDNA5ReadyQueue is built on this. Time advances only on the real clock.
// ---------------------------------------------------------------------------

TEST(InFlightQueue, IssueCapAdmitsNBackToBackThenWaits) {
    InFlightQueue cap(/*depth=*/3);
    const int span = 8;

    // A ceiling, not a quota: all 3 may go with no spacing at all.
    for (int i = 0; i < 3; ++i) {
        EXPECT_FALSE(cap.full()) << "issue " << i << " is within the cap";
        cap.push(span);
    }
    EXPECT_TRUE(cap.full()) << "the 4th must wait";
    EXPECT_EQ(cap.minResidual(), span) << "until the oldest falls out of the window";
}

TEST(InFlightQueue, IssueCapSlidesRatherThanResetting) {
    InFlightQueue cap(/*depth=*/3);
    const int span = 8;
    for (int i = 0; i < 3; ++i) cap.push(span);
    ASSERT_TRUE(cap.full());

    // Half a span of real work: nothing has aged out yet.
    cap.advance(4);
    EXPECT_TRUE(cap.full()) << "a sliding window does not reset part-way";

    // The rest of the span retires all three at once (they issued together).
    cap.advance(4);
    EXPECT_FALSE(cap.full());
    EXPECT_EQ(cap.size(), 0);
}

TEST(InFlightQueue, IssueCapNeverAdmits2NAcrossAWindowEdge) {
    InFlightQueue cap(/*depth=*/3);
    const int span = 8;
    cap.push(span);
    cap.advance(7);  // late in the window
    cap.push(span);
    cap.push(span);
    ASSERT_TRUE(cap.full());

    // A tumbling window anchored on a WMMA issue would reset here and admit a
    // fresh 3, putting 6 back-to-back with no drain between them. A sliding
    // window cannot: the two entries issued at the edge are still inside it, so
    // exactly one slot frees and exactly one more ds_load may go.
    cap.advance(1);
    EXPECT_EQ(cap.size(), 2) << "only the entry from t=0 aged out";
    EXPECT_FALSE(cap.full()) << "one slot freed";
    cap.push(span);
    EXPECT_TRUE(cap.full()) << "and only one: the edge entries still count, so this is 3 in the "
                               "window, never 2N back-to-back";
}

TEST(InFlightQueue, IssueCapSeedsAcrossRegionBoundary) {
    InFlightQueue next(/*depth=*/3);
    next.seed(/*count=*/3, /*residual=*/5);
    EXPECT_TRUE(next.full()) << "a carried-full window must still be full";
    EXPECT_EQ(next.minResidual(), 5);

    InFlightQueue noCarry(/*depth=*/3);
    EXPECT_FALSE(noCarry.full()) << "without the carry the window starts empty";
}

// ---------------------------------------------------------------------------
// Per-entry residual carry (residuals() / seed(vector<int>)): a cross-BB carry
// that keeps each credit's own remaining drain latency instead of collapsing
// every entry to one worst-case value.
// ---------------------------------------------------------------------------

TEST(InFlightQueue, ResidualsReportsEachEntrysOwnRemainingLatency) {
    InFlightQueue queue(/*depth=*/0);
    queue.push(/*drainLatency=*/2);
    queue.push(/*drainLatency=*/4);
    queue.push(/*drainLatency=*/10);

    EXPECT_EQ(queue.residuals(), (std::vector<int>{2, 4, 10}))
        << "oldest-first, matching push order, not sorted or collapsed to one value";
}

TEST(InFlightQueue, SeedFromResidualsRecreatesEachEntrysOwnLatency) {
    InFlightQueue prev(/*depth=*/0);
    prev.push(/*drainLatency=*/2);
    prev.push(/*drainLatency=*/4);
    prev.push(/*drainLatency=*/10);

    InFlightQueue next(/*depth=*/3);
    next.seed(prev.residuals());

    // A (count, worst-case) seed would have reported minResidual == maxResidual
    // == 10 for all three; the per-entry carry preserves the spread instead.
    EXPECT_EQ(next.size(), 3);
    EXPECT_EQ(next.minResidual(), 2);
    EXPECT_EQ(next.maxResidual(), 10);
}

TEST(InFlightQueue, SeedFromResidualsDrainsEntriesAtTheirOwnTimeNotTheWorstCase) {
    InFlightQueue next(/*depth=*/3);
    next.seed(std::vector<int>{2, 4, 10});
    ASSERT_TRUE(next.full());

    // The (count, worst-case) form would keep all three around for 10 cycles;
    // the shortest-lived carried entry must retire on its own schedule.
    next.advance(2);
    EXPECT_EQ(next.size(), 2) << "the entry seeded with residual 2 has drained";
    EXPECT_FALSE(next.full());
}

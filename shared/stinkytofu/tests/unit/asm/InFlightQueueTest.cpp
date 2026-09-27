// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "transforms/asm/dag/InFlightQueue.hpp"

using namespace stinkytofu;

TEST(InFlightQueue, ThrottleTransitionUsesHalfIntervalForOneQueueDepth) {
    InFlightQueue queue(/*depth=*/4);
    queue.setThrottleInterval(/*issueInterval=*/4.0, /*transitionFactor=*/0.5,
                              /*transitionEntries=*/4);

    for (int i = 0; i < 3; ++i) {
        queue.pushWithThrottle(/*drainLatency=*/100);
        EXPECT_EQ(queue.throttleWait(), 0);
    }

    // Reaching depth starts the transition at half the real interval.
    queue.pushWithThrottle(/*drainLatency=*/100);
    EXPECT_EQ(queue.throttleWait(), 2);

    // Three more pushes remain in the transition range (occupancies 5, 6, and 7).
    for (int i = 0; i < 3; ++i) {
        queue.advance(queue.throttleWait());
        queue.pushWithThrottle(/*drainLatency=*/100);
        EXPECT_EQ(queue.throttleWait(), 2);
    }

    // The fourth extra push reaches 2 * depth and enables the real interval.
    queue.advance(queue.throttleWait());
    queue.pushWithThrottle(/*drainLatency=*/100);
    EXPECT_EQ(queue.size(), 8);
    EXPECT_EQ(queue.throttleWait(), 4);
}

TEST(InFlightQueue, DefaultConfigurationHasNoTransitionRange) {
    InFlightQueue queue(/*depth=*/2);
    queue.setThrottleInterval(/*issueInterval=*/4.0);

    queue.pushWithThrottle(/*drainLatency=*/100);
    EXPECT_EQ(queue.throttleWait(), 0);

    queue.pushWithThrottle(/*drainLatency=*/100);
    EXPECT_EQ(queue.throttleWait(), 4);
}

TEST(InFlightQueue, AdvancingThrottleDoesNotDrainEntries) {
    InFlightQueue queue(/*depth=*/1);
    queue.setThrottleInterval(/*issueInterval=*/4.0);

    queue.pushWithThrottle(/*drainLatency=*/10);
    ASSERT_EQ(queue.throttleWait(), 4);
    ASSERT_EQ(queue.minResidual(), 10);

    queue.advanceThrottle(/*cycles=*/4);
    EXPECT_EQ(queue.throttleWait(), 0);
    EXPECT_EQ(queue.minResidual(), 10);
    EXPECT_EQ(queue.size(), 1);

    queue.advance(/*cycles=*/10);
    EXPECT_TRUE(queue.empty());
}

// ---------------------------------------------------------------------------
// Rule (4) ds_load cap, as CDNA5ReadyQueue composes it: a second InFlightQueue
// whose depth is the ceiling N and whose entry lifetime is the window span, so
// full() means "N already issued within the last span cycles".
//
// These cover the window semantics the cap is built on. The scheduler currently
// keeps the cap ANCHORED (cleared on WMMA issue, span set past any window
// length), so only full() and clear() are live there and this commit changes no
// behaviour. Span-based retirement -- everything below about sliding -- becomes
// live when the cap is unanchored, which is the follow-up; these tests pin the
// semantics that change will depend on.
//
// Throughout: the cap rides the real clock (advance), never the throttle clock,
// because only real cycles age entries out of a queue.
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

TEST(InFlightQueue, IssueCapIgnoresThrottleOnlyTime) {
    InFlightQueue cap(/*depth=*/2);
    const int span = 8;
    cap.push(span);
    cap.push(span);
    ASSERT_TRUE(cap.full());

    // Throttle debt is pacing bookkeeping, not elapsed execution: it ages
    // nothing, so it must not hand out cap allowance either.
    cap.advanceThrottle(64);
    EXPECT_TRUE(cap.full())
        << "only real elapsed cycles may retire a cap entry -- evict() compares "
           "against currentTime_, which advanceThrottle() does not move";
}

TEST(InFlightQueue, IssueCapSeedsAcrossRegionBoundary) {
    InFlightQueue next(/*depth=*/3);
    next.seed(/*count=*/3, /*residual=*/5);
    EXPECT_TRUE(next.full()) << "a carried-full window must still be full";
    EXPECT_EQ(next.minResidual(), 5);

    InFlightQueue noCarry(/*depth=*/3);
    EXPECT_FALSE(noCarry.full()) << "without the carry the window starts empty";
}

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

// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "RingPolicy.hpp"

#include <gtest/gtest.h>

#include <cstddef>

using TensileLite::Client::RingPolicy;
using TensileLite::Client::RingPolicyInputs;
using TensileLite::Client::benchmarkEnqueuesMayExecuteIfSolutionRuns;
using TensileLite::Client::benchmarkTimerRequestsSolutionRuns;
using TensileLite::Client::chooseRingPolicy;
using TensileLite::Client::effectiveEnqueuesMayBePositive;
using TensileLite::Client::hasValidationDriver;

namespace
{
    RingPolicyInputs makeInputs(int    numBenchmarks,
                                int    numEnqueuesPerSync,
                                int    maxEnqueuesPerSync,
                                int    numSyncsPerBenchmark,
                                size_t minFlopsPerSync,
                                int    numElementsToValidate,
                                bool   printAny)
    {
        return {numBenchmarks,
                numEnqueuesPerSync,
                maxEnqueuesPerSync,
                numSyncsPerBenchmark,
                minFlopsPerSync,
                numElementsToValidate,
                printAny};
    }

    void expectDisabledPolicy(RingPolicy const& policy)
    {
        EXPECT_FALSE(policy.allowed);
        EXPECT_EQ(policy.activeBufferCount, 1u);
        EXPECT_FALSE(policy.allocatesAltBuffers());
    }

    void expectAllowedPolicy(RingPolicy const& policy)
    {
        EXPECT_TRUE(policy.allowed);
        EXPECT_EQ(policy.activeBufferCount, 3u);
        EXPECT_TRUE(policy.allocatesAltBuffers());
    }
} // namespace

TEST(RingPolicy, TimedBenchmarkDisablesRing)
{
    auto inputs = makeInputs(1, 1, -1, 1, 0, 0, false);

    EXPECT_FALSE(hasValidationDriver(inputs));
    EXPECT_TRUE(benchmarkTimerRequestsSolutionRuns(inputs));
    EXPECT_TRUE(benchmarkEnqueuesMayExecuteIfSolutionRuns(inputs));

    expectDisabledPolicy(chooseRingPolicy(inputs));
}

TEST(RingPolicy, TimedBenchmarkWinsOverValidation)
{
    auto inputs = makeInputs(1, 1, -1, 1, 0, 1, false);

    EXPECT_TRUE(hasValidationDriver(inputs));
    EXPECT_TRUE(benchmarkTimerRequestsSolutionRuns(inputs));
    EXPECT_TRUE(benchmarkEnqueuesMayExecuteIfSolutionRuns(inputs));

    expectDisabledPolicy(chooseRingPolicy(inputs));
}

TEST(RingPolicy, ValidationWithPositiveEnqueueSyncCountersDisablesRing)
{
    auto inputs = makeInputs(0, 1, -1, 1, 0, 1, false);

    EXPECT_TRUE(hasValidationDriver(inputs));
    EXPECT_TRUE(benchmarkTimerRequestsSolutionRuns(inputs));
    EXPECT_TRUE(benchmarkEnqueuesMayExecuteIfSolutionRuns(inputs));

    expectDisabledPolicy(chooseRingPolicy(inputs));
}

TEST(RingPolicy, PrintWithPositiveEnqueueSyncCountersDisablesRing)
{
    auto inputs = makeInputs(0, 1, -1, 1, 0, 0, true);

    EXPECT_TRUE(hasValidationDriver(inputs));
    EXPECT_TRUE(benchmarkTimerRequestsSolutionRuns(inputs));
    EXPECT_TRUE(benchmarkEnqueuesMayExecuteIfSolutionRuns(inputs));

    expectDisabledPolicy(chooseRingPolicy(inputs));
}

TEST(RingPolicy, ValidationWithMinFlopsEffectiveEnqueuesDisablesRing)
{
    auto inputs = makeInputs(0, 0, -1, 1, 1, 1, false);

    EXPECT_TRUE(hasValidationDriver(inputs));
    EXPECT_FALSE(benchmarkTimerRequestsSolutionRuns(inputs));
    EXPECT_TRUE(effectiveEnqueuesMayBePositive(inputs));
    EXPECT_TRUE(benchmarkEnqueuesMayExecuteIfSolutionRuns(inputs));

    expectDisabledPolicy(chooseRingPolicy(inputs));
}

TEST(RingPolicy, ValidationWithZeroRawAndNoMinFlopsUsesThreeSlots)
{
    auto inputs = makeInputs(0, 0, -1, 1, 0, 1, false);

    EXPECT_TRUE(hasValidationDriver(inputs));
    EXPECT_FALSE(benchmarkTimerRequestsSolutionRuns(inputs));
    EXPECT_FALSE(effectiveEnqueuesMayBePositive(inputs));
    EXPECT_FALSE(benchmarkEnqueuesMayExecuteIfSolutionRuns(inputs));

    expectAllowedPolicy(chooseRingPolicy(inputs));
}

TEST(RingPolicy, ValidationWithZeroSyncUsesThreeSlots)
{
    auto inputs = makeInputs(0, 1, -1, 0, 0, 1, false);

    EXPECT_TRUE(hasValidationDriver(inputs));
    EXPECT_FALSE(benchmarkTimerRequestsSolutionRuns(inputs));
    EXPECT_FALSE(benchmarkEnqueuesMayExecuteIfSolutionRuns(inputs));

    expectAllowedPolicy(chooseRingPolicy(inputs));
}

TEST(RingPolicy, ValidationWithMaxEnqueuesZeroUsesThreeSlots)
{
    auto inputs = makeInputs(0, 0, 0, 1, 1, 1, false);

    EXPECT_TRUE(hasValidationDriver(inputs));
    EXPECT_FALSE(benchmarkTimerRequestsSolutionRuns(inputs));
    EXPECT_FALSE(effectiveEnqueuesMayBePositive(inputs));
    EXPECT_FALSE(benchmarkEnqueuesMayExecuteIfSolutionRuns(inputs));

    expectAllowedPolicy(chooseRingPolicy(inputs));
}

TEST(RingPolicy, ValidationWithRawBenchmarkRequestAndMaxZeroDisablesRing)
{
    auto inputs = makeInputs(0, 1, 0, 1, 0, 1, false);

    EXPECT_TRUE(hasValidationDriver(inputs));
    EXPECT_TRUE(benchmarkTimerRequestsSolutionRuns(inputs));
    EXPECT_FALSE(effectiveEnqueuesMayBePositive(inputs));
    EXPECT_FALSE(benchmarkEnqueuesMayExecuteIfSolutionRuns(inputs));

    expectDisabledPolicy(chooseRingPolicy(inputs));
}

TEST(RingPolicy, PrintOnlyUntimedUsesThreeSlots)
{
    auto inputs = makeInputs(0, 0, -1, 1, 0, 0, true);

    EXPECT_TRUE(hasValidationDriver(inputs));
    EXPECT_FALSE(benchmarkTimerRequestsSolutionRuns(inputs));
    EXPECT_FALSE(effectiveEnqueuesMayBePositive(inputs));
    EXPECT_FALSE(benchmarkEnqueuesMayExecuteIfSolutionRuns(inputs));

    expectAllowedPolicy(chooseRingPolicy(inputs));
}

TEST(RingPolicy, ZeroEnqueueNoValidationDisablesRing)
{
    auto inputs = makeInputs(0, 0, -1, 0, 0, 0, false);

    EXPECT_FALSE(hasValidationDriver(inputs));
    EXPECT_FALSE(benchmarkTimerRequestsSolutionRuns(inputs));
    EXPECT_FALSE(effectiveEnqueuesMayBePositive(inputs));
    EXPECT_FALSE(benchmarkEnqueuesMayExecuteIfSolutionRuns(inputs));

    expectDisabledPolicy(chooseRingPolicy(inputs));
}

TEST(RingPolicy, PartialBenchmarkCountersWithoutValidationDisableRing)
{
    auto inputs = makeInputs(1, 0, -1, 1, 0, 0, false);

    EXPECT_FALSE(hasValidationDriver(inputs));
    EXPECT_FALSE(benchmarkTimerRequestsSolutionRuns(inputs));
    EXPECT_FALSE(benchmarkEnqueuesMayExecuteIfSolutionRuns(inputs));

    expectDisabledPolicy(chooseRingPolicy(inputs));
}

// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "BenchmarkTimer.hpp"

#include <cstddef>

using namespace TensileLite::Client;

namespace
{
    po::variables_map timingArgs(int syncs, int enqueues, int maxEnqueues, size_t minFlops)
    {
        po::variables_map args;
        args["num-syncs-per-benchmark"].value() = syncs;
        args["num-enqueues-per-sync"].value()   = enqueues;
        args["max-enqueues-per-sync"].value()   = maxEnqueues;
        args["min-flops-per-sync"].value()      = minFlops;
        return args;
    }

    po::variables_map warmupArgs(int syncs, float skipSlowRatio, bool syncAfterWarmups)
    {
        po::variables_map args                   = timingArgs(syncs, 1, -1, 1);
        args["skip-slow-solution-ratio"].value() = skipSlowRatio;
        args["sync-after-warmups"].value()       = syncAfterWarmups;
        return args;
    }
} // namespace

// The four syncs/enqueues/min-flops combinations measured in
// timing_runs/per_solution_bookkeeping_investigation.md, "Deciding when a run is a test"
// (max-enqueues-per-sync at its default -1, no cap). Zero enqueues alone still time: the
// flops term asks for at least one.
TEST(BenchmarkTimerNothingIsTimed, SyncsEnqueuesMinFlops)
{
    EXPECT_TRUE(BenchmarkTimer::nothingIsTimed(timingArgs(0, 1, -1, 1)));
    EXPECT_FALSE(BenchmarkTimer::nothingIsTimed(timingArgs(1, 0, -1, 1)));
    EXPECT_TRUE(BenchmarkTimer::nothingIsTimed(timingArgs(1, 0, -1, 0)));
    EXPECT_FALSE(BenchmarkTimer::nothingIsTimed(timingArgs(1, 1, -1, 1)));
}

// A zero enqueue cap rules out timing on its own; a negative count or cap casts to a huge
// size_t in numSyncs()/numEnqueuesPerSync(), so it times.
TEST(BenchmarkTimerNothingIsTimed, CapsAndNegativeValues)
{
    EXPECT_TRUE(BenchmarkTimer::nothingIsTimed(timingArgs(1, 1, 0, 1)));
    EXPECT_FALSE(BenchmarkTimer::nothingIsTimed(timingArgs(1, 1, 1, 0)));
    EXPECT_FALSE(BenchmarkTimer::nothingIsTimed(timingArgs(-1, 1, -1, 0)));
    EXPECT_FALSE(BenchmarkTimer::nothingIsTimed(timingArgs(1, -1, -1, 0)));
}

// Warmup events are dropped only when nothing is timed, the skip-slow ratio is 0 and
// sync-after-warmups is off; any one reader keeps them.
TEST(BenchmarkTimerReadsWarmupEvents, NothingTimedSkipSlowSyncAfterWarmups)
{
    EXPECT_FALSE(BenchmarkTimer::readsWarmupEvents(warmupArgs(0, 0.0f, false)));
    EXPECT_TRUE(BenchmarkTimer::readsWarmupEvents(warmupArgs(0, 0.5f, false)));
    EXPECT_TRUE(BenchmarkTimer::readsWarmupEvents(warmupArgs(0, 0.0f, true)));
    EXPECT_TRUE(BenchmarkTimer::readsWarmupEvents(warmupArgs(1, 0.0f, false)));
    EXPECT_TRUE(BenchmarkTimer::readsWarmupEvents(warmupArgs(1, 0.5f, true)));
}

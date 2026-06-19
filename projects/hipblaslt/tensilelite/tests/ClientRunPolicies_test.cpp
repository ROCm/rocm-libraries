// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ClientRunPolicies.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

using namespace TensileLite::Client;

namespace
{
    IcacheAutoRotationPlan computeIcachePlan(
        size_t                          inputSlotCount,
        std::vector<std::string> const& codeObjectFilenames,
        int                             rotateSizeKB,
        KernelHotPathSizeFn const&      kernelHotPathSizeFn)
    {
        IcacheRotationPolicy policy;
        return policy.computeAutoPlan(inputSlotCount,
                                      codeObjectFilenames,
                                      rotateSizeKB,
                                      kernelHotPathSizeFn);
    }
} // namespace

TEST(RotatingOutputPolicy, WarmupDominatesBenchmarkCount)
{
    RotatingOutputPolicy policy;
    auto plan = policy.plan(7, 2, 3);

    EXPECT_EQ(plan.warmupRuns, 7u);
    EXPECT_EQ(plan.syncs, 2u);
    EXPECT_EQ(plan.enqueuesPerSync, 3u);
    EXPECT_EQ(plan.maxRotatingBufferNum, 7);
}

TEST(RotatingOutputPolicy, BenchmarkCountDominatesWarmupCount)
{
    RotatingOutputPolicy policy;
    auto plan = policy.plan(1, 2, 3);

    EXPECT_EQ(plan.warmupRuns, 1u);
    EXPECT_EQ(plan.syncs, 2u);
    EXPECT_EQ(plan.enqueuesPerSync, 3u);
    EXPECT_EQ(plan.maxRotatingBufferNum, 6);
}

TEST(RotatingOutputPolicy, ZeroSyncUsesWarmupCount)
{
    RotatingOutputPolicy policy;
    auto plan = policy.plan(5, 0, 4);

    EXPECT_EQ(plan.warmupRuns, 5u);
    EXPECT_EQ(plan.syncs, 0u);
    EXPECT_EQ(plan.enqueuesPerSync, 4u);
    EXPECT_EQ(plan.maxRotatingBufferNum, 5);
}

TEST(RotatingOutputPolicy, ZeroEnqueueUsesWarmupCount)
{
    RotatingOutputPolicy policy;
    auto plan = policy.plan(5, 4, 0);

    EXPECT_EQ(plan.warmupRuns, 5u);
    EXPECT_EQ(plan.syncs, 4u);
    EXPECT_EQ(plan.enqueuesPerSync, 0u);
    EXPECT_EQ(plan.maxRotatingBufferNum, 5);
}

TEST(RotatingOutputPolicy, ZeroEnqueueAndNoWarmupUsesZeroBuffers)
{
    RotatingOutputPolicy policy;
    auto plan = policy.plan(0, 2, 0);

    EXPECT_EQ(plan.warmupRuns, 0u);
    EXPECT_EQ(plan.syncs, 2u);
    EXPECT_EQ(plan.enqueuesPerSync, 0u);
    EXPECT_EQ(plan.maxRotatingBufferNum, 0);
}

TEST(IcacheRotationPolicy, CacheTermDominatesDataInitCount)
{
    auto plan = computeIcachePlan(2,
                                   {"a.co"},
                                   2,
                                   [](std::string const&) -> std::uintmax_t {
                                       return 512;
                                   });

    EXPECT_EQ(plan.extrasFromDataInit, 1);
#if defined(__linux__)
    EXPECT_EQ(plan.kernelHotPathSize, 512u);
    EXPECT_EQ(plan.cacheBudgetBytes, 4096u);
    EXPECT_EQ(plan.extrasFromCache, 7);
#else
    EXPECT_EQ(plan.kernelHotPathSize, 0u);
    EXPECT_EQ(plan.cacheBudgetBytes, 0u);
    EXPECT_EQ(plan.extrasFromCache, 2);
#endif
    EXPECT_EQ(plan.extras, plan.extrasFromCache);
    EXPECT_GT(plan.extrasFromCache, plan.extrasFromDataInit);
}

TEST(IcacheRotationPolicy, PositiveInputSlotsNoPositiveHotPathUsesDataInitCount)
{
    auto plan = computeIcachePlan(3,
                                   {"a.co"},
                                   64,
                                   [](std::string const&) -> std::uintmax_t {
                                       return 0;
                                   });

    EXPECT_EQ(plan.extrasFromDataInit, 2);
#if defined(__linux__)
    EXPECT_EQ(plan.kernelHotPathSize, 0u);
    EXPECT_EQ(plan.cacheBudgetBytes, 131072u);
    EXPECT_EQ(plan.extrasFromCache, 0);
    EXPECT_EQ(plan.extras, 2);
#else
    EXPECT_EQ(plan.kernelHotPathSize, 0u);
    EXPECT_EQ(plan.cacheBudgetBytes, 0u);
    EXPECT_EQ(plan.extrasFromCache, 64);
    EXPECT_EQ(plan.extras, 64);
#endif
}

TEST(IcacheRotationPolicy, ZeroInputSlotsCastBeforeSubtract)
{
    auto plan = computeIcachePlan(0, {}, 64, KernelHotPathSizeFn{});

    EXPECT_EQ(plan.extrasFromDataInit, -1);
#if defined(__linux__)
    EXPECT_EQ(plan.extrasFromCache, 0);
    EXPECT_EQ(plan.extras, 0);
    EXPECT_EQ(plan.kernelHotPathSize, 0u);
    EXPECT_EQ(plan.cacheBudgetBytes, 131072u);
#else
    EXPECT_EQ(plan.extrasFromCache, 64);
    EXPECT_EQ(plan.extras, 64);
    EXPECT_EQ(plan.kernelHotPathSize, 0u);
    EXPECT_EQ(plan.cacheBudgetBytes, 0u);
#endif
}

TEST(IcacheRotationPolicy, ShouldLoadAutoCopiesRequiresAutoRequestAndSingleModule)
{
    IcacheRotationPolicy policy;

    EXPECT_TRUE(policy.shouldLoadAutoCopies(-1, 1));
    EXPECT_FALSE(policy.shouldLoadAutoCopies(0, 1));
    EXPECT_FALSE(policy.shouldLoadAutoCopies(-1, 2));
    EXPECT_FALSE(policy.shouldLoadAutoCopies(0, 2));
}

TEST(IcacheRotationCursor, CyclesAndResets)
{
    IcacheRotationCursor cursor;

    EXPECT_EQ(cursor.nextIndex(3), 0);
    EXPECT_EQ(cursor.nextIndex(3), 1);
    EXPECT_EQ(cursor.nextIndex(3), 2);
    EXPECT_EQ(cursor.nextIndex(3), 0);

    cursor.reset();

    EXPECT_EQ(cursor.nextIndex(2), 0);
}

// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>

#include "transforms/asm/dag/DsPacePlan.hpp"

using namespace stinkytofu;

namespace {

int countIn(const std::vector<int>& plan, int begin, int end) {
    return static_cast<int>(
        std::count_if(plan.begin(), plan.end(), [&](int t) { return t >= begin && t < end; }));
}

}  // namespace

// No windows: a uniform spread of `count` ds_loads over [start, end).
TEST(DsPacePlan, UniformWithoutWindows) {
    const auto plan = planDsRelease(100, 0, 1000, /*maxRate=*/0.5, /*windowRate=*/0.1, {});
    ASSERT_EQ(plan.size(), 100u);
    EXPECT_EQ(plan.front(), 0);
    EXPECT_LE(plan.back(), 1000);
    EXPECT_TRUE(std::is_sorted(plan.begin(), plan.end()));
    // 0.1 ds/cycle: one per 10 cycles.
    EXPECT_EQ(plan[10], 100);
    EXPECT_EQ(plan[50], 500);
}

// The window caps its own rate and the count is kept: the rate outside rises.
TEST(DsPacePlan, WindowThinnedCountKept) {
    // 200 ds over 2000 cycles = 0.1/cycle; window [800,1200) allows 0.025/cycle.
    const auto plan =
        planDsRelease(200, 0, 2000, /*maxRate=*/0.25, /*windowRate=*/0.025, {{800, 1200}});
    ASSERT_EQ(plan.size(), 200u);
    // Inside the window: at most 0.025 * 400 = 10 ds (plus rounding).
    EXPECT_LE(countIn(plan, 800, 1200), 11);
    // Outside the window the rate is (200 - 10) / 1600 = 0.11875/cycle -- higher than the uniform
    // 0.1.
    EXPECT_GT(countIn(plan, 0, 800), 80);
    EXPECT_TRUE(std::is_sorted(plan.begin(), plan.end()));
    EXPECT_LE(plan.back(), 2000);
}

// A window the uniform rate already satisfies does not bind: same plan as without it.
TEST(DsPacePlan, NonBindingWindowChangesNothing) {
    const auto base = planDsRelease(100, 0, 1000, 0.5, 0.2, {});
    const auto withWin = planDsRelease(100, 0, 1000, 0.5, 0.2, {{300, 500}});
    EXPECT_EQ(base, withWin);
}

// Not feasible under the ceiling or an empty range: no plan (the scheduler stays ASAP).
TEST(DsPacePlan, InfeasibleGivesNoPlan) {
    // Needs an outside rate of (200 - 0)/ (1000 - 800) = 1.0 > maxRate 0.5.
    EXPECT_TRUE(planDsRelease(200, 0, 1000, 0.5, 0.0, {{200, 1000}}).empty());
    EXPECT_TRUE(planDsRelease(0, 0, 1000, 0.5, 0.1, {}).empty());
    EXPECT_TRUE(planDsRelease(10, 500, 500, 0.5, 0.1, {}).empty());
    EXPECT_TRUE(planDsRelease(10, 0, 100, 0.0, 0.1, {}).empty());
}

// Windows are clipped to the range and overlapping ones merged.
TEST(DsPacePlan, WindowsClippedAndMerged) {
    const auto a = planDsRelease(100, 0, 1000, 0.5, 0.02, {{-100, 100}, {900, 2000}, {90, 150}});
    ASSERT_EQ(a.size(), 100u);
    EXPECT_LE(countIn(a, 0, 150), 4);  // merged window [0,150) at 0.02/cycle = 3 ds
    EXPECT_LE(countIn(a, 900, 1000), 3);
    EXPECT_TRUE(std::is_sorted(a.begin(), a.end()));
}

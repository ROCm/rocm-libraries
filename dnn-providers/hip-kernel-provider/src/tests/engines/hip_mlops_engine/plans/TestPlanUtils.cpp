// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <climits>
#include <cstdint>
#include <gtest/gtest.h>

#include "engines/hip_mlops_engine/plans/PlanUtils.hpp"
#include "hipdnn_plugin_sdk/PluginException.hpp"

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::batchnorm;

TEST(TestComputeVectorSize, ChannelsDivisibleByFour)
{
    EXPECT_EQ(computeVectorSize(true, 16, 7), 4);
}

TEST(TestComputeVectorSize, ChannelsDivisibleByTwo)
{
    EXPECT_EQ(computeVectorSize(true, 6, 8), 2);
}

TEST(TestComputeVectorSize, ChannelsAreOdd)
{
    EXPECT_EQ(computeVectorSize(true, 3, 4), 1);
}

TEST(TestComputeVectorSize, StrideDivisibleByFour)
{
    EXPECT_EQ(computeVectorSize(false, 3, 8), 4);
}

TEST(TestComputeVectorSize, StrideDivisibleByTwo)
{
    EXPECT_EQ(computeVectorSize(false, 16, 10), 2);
}

TEST(TestComputeVectorSize, StrideIsOdd)
{
    EXPECT_EQ(computeVectorSize(false, 8, 7), 1);
}

TEST(TestCheckedNarrowToInt, PositiveValueInRange)
{
    EXPECT_EQ(checkedNarrowToInt(42, "test"), 42);
}

TEST(TestCheckedNarrowToInt, Zero)
{
    EXPECT_EQ(checkedNarrowToInt(0, "test"), 0);
}

TEST(TestCheckedNarrowToInt, NegativeValueInRange)
{
    EXPECT_EQ(checkedNarrowToInt(-100, "test"), -100);
}

TEST(TestCheckedNarrowToInt, IntMaxBoundary)
{
    auto maxVal = static_cast<int64_t>(INT_MAX);
    EXPECT_EQ(checkedNarrowToInt(maxVal, "test"), INT_MAX);
}

TEST(TestCheckedNarrowToInt, IntMinBoundary)
{
    auto minVal = static_cast<int64_t>(INT_MIN);
    EXPECT_EQ(checkedNarrowToInt(minVal, "test"), INT_MIN);
}

TEST(TestCheckedNarrowToInt, ExceedsIntMaxThrows)
{
    auto overMax = static_cast<int64_t>(INT_MAX) + 1;
    EXPECT_THROW(checkedNarrowToInt(overMax, "dim"), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestCheckedNarrowToInt, BelowIntMinThrows)
{
    auto underMin = static_cast<int64_t>(INT_MIN) - 1;
    EXPECT_THROW(checkedNarrowToInt(underMin, "dim"), hipdnn_plugin_sdk::HipdnnPluginException);
}

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
    const auto maxVal = static_cast<int64_t>(INT_MAX);
    EXPECT_EQ(checkedNarrowToInt(maxVal, "test"), INT_MAX);
}

TEST(TestCheckedNarrowToInt, IntMinBoundary)
{
    const auto minVal = static_cast<int64_t>(INT_MIN);
    EXPECT_EQ(checkedNarrowToInt(minVal, "test"), INT_MIN);
}

TEST(TestCheckedNarrowToInt, ExceedsIntMaxThrows)
{
    const auto overMax = static_cast<int64_t>(INT_MAX) + 1;
    EXPECT_THROW(checkedNarrowToInt(overMax, "dim"), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestCheckedNarrowToInt, BelowIntMinThrows)
{
    const auto underMin = static_cast<int64_t>(INT_MIN) - 1;
    EXPECT_THROW(checkedNarrowToInt(underMin, "dim"), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestLaunchFitsThreadLimit, BlocksTimesThreadsAtUint32MaxFits)
{
    EXPECT_TRUE(launchFitsThreadLimit(4194303, 1024));
    EXPECT_TRUE(launchFitsThreadLimit(16777215, 256));
    EXPECT_TRUE(launchFitsThreadLimit(1, UINT32_MAX));
}

TEST(TestLaunchFitsThreadLimit, BlocksTimesThreadsReaching2To32DoesNotFit)
{
    EXPECT_FALSE(launchFitsThreadLimit(4194304, 1024));
    EXPECT_FALSE(launchFitsThreadLimit(16777216, 256));
    EXPECT_FALSE(launchFitsThreadLimit(2, int64_t{1} << 31));
}

TEST(TestLaunchFitsThreadLimit, EmptyLaunchDoesNotFit)
{
    EXPECT_FALSE(launchFitsThreadLimit(0, 256));
    EXPECT_FALSE(launchFitsThreadLimit(256, 0));
}

TEST(TestLaunchFitsThreadLimit, ProductOverflowingInt64DoesNotFit)
{
    EXPECT_FALSE(launchFitsThreadLimit(INT64_MAX, 1024));
}

TEST(TestCheckLaunchFitsThreadLimit, ThrowsWhenLaunchDoesNotFit)
{
    EXPECT_THROW(checkLaunchFitsThreadLimit({"kernel", 4194304, 1024}),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestCheckLaunchFitsThreadLimit, DoesNotThrowWhenLaunchFits)
{
    EXPECT_NO_THROW(checkLaunchFitsThreadLimit({"kernel", 4194303, 1024}));
}

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "engines/hip_mlops_engine/plans/PlanUtils.hpp"

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

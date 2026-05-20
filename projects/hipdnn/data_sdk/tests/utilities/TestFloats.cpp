// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types.hpp>

// Import the custom types for use in tests
using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::half;

// Import user-defined literals for convenience
using namespace hipdnn_data_sdk::types;

TEST(TestUtilsFp16, BasicUsage)
{
    const half h = 1.0_h;
    EXPECT_EQ(h, 1.0_h);
}

TEST(TestUtilsFp16, Fabs)
{
    EXPECT_EQ(fabs(-1.0_h), 1.0_h);
    EXPECT_EQ(fabs(1.0_h), 1.0_h);
}

TEST(TestUtilsFp16, Comparison)
{
    EXPECT_EQ(1.25_h, 1.25_h);
    ASSERT_NE(1.0_h, 2.0_h);
    ASSERT_LT(0.0156_h, 0.0176_h);
    ASSERT_LE(-0.0156_h, 0.0176_h);
    ASSERT_GT(0.0215_h, 0.0176_h);
    ASSERT_GE(0.0215_h, -0.0176_h);
}

TEST(TestUtilsFp16, Max)
{
    const half a = 1.0_h;
    const half b = 2.0_h;
    EXPECT_EQ(max(a, b), 2.0_h);
    EXPECT_EQ(max(b, a), 2.0_h);
}

TEST(TestUtilsBfp16, BasicUsage)
{
    const bfloat16 bf = 1.0_bf;
    EXPECT_EQ(bf, 1.0_bf);
}

TEST(TestUtilsBfp16, Fabs)
{
    EXPECT_EQ(fabs(-1.0_bf), 1.0_bf);
    EXPECT_EQ(fabs(1.0_bf), 1.0_bf);
}

TEST(TestUtilsBfp16, Comparison)
{
    EXPECT_EQ(1.25_bf, 1.25_bf);
    ASSERT_NE(1.0_bf, 2.0_bf);
    ASSERT_LT(0.017_bf, 0.018_bf);
    ASSERT_LE(-0.017_bf, 0.018_bf);
    ASSERT_GT(0.022_bf, 0.018_bf);
    ASSERT_GE(0.022_bf, -0.018_bf);
}

TEST(TestUtilsBfp16, Max)
{
    const bfloat16 a = 1.0_bf;
    const bfloat16 b = 2.0_bf;
    EXPECT_EQ(max(a, b), 2.0_bf);
    EXPECT_EQ(max(b, a), 2.0_bf);
}

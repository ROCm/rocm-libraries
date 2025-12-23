// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_data_sdk/utilities/UtilsFp16.hpp>
#include <hipdnn_data_sdk/utilities/UtilsFp8.hpp>

TEST(TestUtilsFp16, BasicUsage)
{
    half h = 1.0_h;
    EXPECT_EQ(h, 1.0_h);
}

TEST(TestUtilsFp16, Fabs)
{
    EXPECT_EQ(std::fabs(-1.0_h), 1.0_h);
    EXPECT_EQ(std::fabs(1.0_h), 1.0_h);
}

TEST(TestUtilsFp16, Max)
{
    half a = 1.0_h;
    half b = 2.0_h;
    EXPECT_EQ(std::max(a, b), 2.0_h);
    EXPECT_EQ(std::max(b, a), 2.0_h);
}

TEST(TestUtilsBfp16, BasicUsage)
{
    hip_bfloat16 bf = 1.0_bf;
    EXPECT_EQ(bf, 1.0_bf);
}

TEST(TestUtilsBfp16, Fabs)
{
    EXPECT_EQ(std::fabs(-1.0_bf), 1.0_bf);
    EXPECT_EQ(std::fabs(1.0_bf), 1.0_bf);
}

TEST(TestUtilsBfp16, Max)
{
    hip_bfloat16 a = 1.0_bf;
    hip_bfloat16 b = 2.0_bf;
    EXPECT_EQ(std::max(a, b), 2.0_bf);
    EXPECT_EQ(std::max(b, a), 2.0_bf);
}

TEST(TestUtilsFp8, BasicUsage)
{
    hip_fp8_e4m3 fp8 = 1.0_fp8;
    EXPECT_EQ(fp8, 1.0_fp8);
}

TEST(TestUtilsFp8, Fabs)
{
    EXPECT_EQ(std::fabs(-1.0_fp8), 1.0_fp8);
    EXPECT_EQ(std::fabs(1.0_fp8), 1.0_fp8);
}

TEST(TestUtilsFp8, Max)
{
    hip_fp8_e4m3 a = 1.0_fp8;
    hip_fp8_e4m3 b = 2.0_fp8;
    hip_fp8_e4m3 nan = hipdnn_sdk::utilities::fp8::uchar_as_fp8(0x7F);
    EXPECT_EQ(std::max(a, b), 2.0_fp8);
    EXPECT_EQ(std::max(b, a), 2.0_fp8);
    EXPECT_EQ(std::max(a, nan), 1.0_fp8);
    EXPECT_EQ(std::max(nan, b), 2.0_fp8);
    EXPECT_EQ(std::max(nan, nan), nan);
}

// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_data_sdk/utilities/UtilsBfp8.hpp>
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

TEST(TestUtilsBfp8, BasicUsage)
{
    hip_fp8_e5m2 bf = 1.0_bfp8;
    EXPECT_EQ(bf, 1.0_bfp8);
}

TEST(TestUtilsBfp8, Fabs)
{
    EXPECT_EQ(std::fabs(-1.0_bfp8), 1.0_bfp8);
    EXPECT_EQ(std::fabs(1.0_bfp8), 1.0_bfp8);
}

TEST(TestUtilsTestUtilsBfp8, Max)
{
    hip_fp8_e5m2 a = 1.0_bfp8;
    hip_fp8_e5m2 b = 2.0_bfp8;
    hip_fp8_e5m2 nan = hipdnn_sdk::utilities::bfp8::uchar_as_bfp8(0x7F);
    EXPECT_EQ(std::max(a, b), 2.0_bfp8);
    EXPECT_EQ(std::max(b, a), 2.0_bfp8);
    EXPECT_EQ(std::max(a, nan), 1.0_bfp8);
    EXPECT_EQ(std::max(nan, b), 2.0_bfp8);
    EXPECT_EQ(std::max(nan, nan), nan);
}

TEST(TestUtilsFp8E8M0, BasicUsage)
{
    hip_fp8_e8m0 x = 1.0_fp8_e8m0;
    EXPECT_EQ(x, 1.0_fp8_e8m0);
}

TEST(TestUtilsFp8E8M0, Fabs)
{
    hip_fp8_e8m0 a = 1.0_fp8_e8m0;
    EXPECT_EQ(std::fabs(a), a);
    EXPECT_EQ(std::abs(a), a);
}

TEST(TestUtilsFp8E8M0, Max)
{
    hip_fp8_e8m0 a = 1.0_fp8_e8m0;
    hip_fp8_e8m0 b = 2.0_fp8_e8m0;
    hip_fp8_e8m0 nan = HIPDNN_NAN_FP8_E8M0;
    EXPECT_EQ(std::max(a, b), b);
    EXPECT_EQ(std::max(b, a), b);
    EXPECT_EQ(std::max(a, nan), a);
    EXPECT_EQ(std::max(nan, b), b);
    EXPECT_EQ(std::max(nan, nan), nan);
}

TEST(TestUtilsFp8E8M0, CmpOperators)
{
    hip_fp8_e8m0 a = 1.0_fp8_e8m0;
    hip_fp8_e8m0 b = 2.0_fp8_e8m0;
    EXPECT_LT(a, b);
    EXPECT_GT(b, a);
    EXPECT_LE(a, a);
    EXPECT_GE(b, a);
    EXPECT_NE(a, b);
    EXPECT_EQ(a, a);
}

TEST(TestUtilsFp8E8M0, IsNan)
{
    hip_fp8_e8m0 a = 1.0_fp8_e8m0;
    hip_fp8_e8m0 nan = HIPDNN_NAN_FP8_E8M0;
    EXPECT_FALSE(std::isnan(a));
    EXPECT_TRUE(std::isnan(nan));
}

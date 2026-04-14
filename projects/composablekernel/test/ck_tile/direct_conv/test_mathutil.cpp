// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck_tile/ops/direct_convolution/utils/mathutil.hpp"

using namespace ck_tile::direct_conv;

TEST(MathUtil, MaximumBasic)
{
    EXPECT_EQ(maximum(3, 5), 5);
    EXPECT_EQ(maximum(5, 3), 5);
    EXPECT_EQ(maximum(4, 4), 4);
}

TEST(MathUtil, MaximumNegative)
{
    EXPECT_EQ(maximum(-1, -2), -1);
    EXPECT_EQ(maximum(-5, 0), 0);
    EXPECT_EQ(maximum(0, 0), 0);
}

TEST(MathUtil, MaximumConstexpr)
{
    static_assert(maximum(3, 5) == 5);
    static_assert(maximum(5, 3) == 5);
    static_assert(maximum(0, 0) == 0);
}

TEST(MathUtil, DivupBasic)
{
    EXPECT_EQ(divup(10, 3), 4);
    EXPECT_EQ(divup(9, 3), 3);
    EXPECT_EQ(divup(1, 1), 1);
    EXPECT_EQ(divup(0, 5), 0);
}

TEST(MathUtil, DivupExact)
{
    EXPECT_EQ(divup(6, 3), 2);
    EXPECT_EQ(divup(64, 64), 1);
    EXPECT_EQ(divup(128, 64), 2);
}

TEST(MathUtil, DivupConstexpr)
{
    static_assert(divup(10, 3) == 4);
    static_assert(divup(9, 3) == 3);
    static_assert(divup(0, 5) == 0);
}

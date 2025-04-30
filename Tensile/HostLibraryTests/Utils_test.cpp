/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <gtest/gtest.h>

#include <Tensile/Utils.hpp>

using namespace Tensile;

TEST(UtilsTest, CeilDivide)
{
    EXPECT_EQ(CeilDivide(16, 4), 4);
    EXPECT_EQ(CeilDivide(16, 3), 6);

    EXPECT_EQ(CeilDivide(800, 64), 13);
}

TEST(UtilsTest, RoundUpToMultiple)
{
    EXPECT_EQ(RoundUpToMultiple(10, 4), 12);
    EXPECT_EQ(RoundUpToMultiple(12, 4), 12);
    EXPECT_EQ(RoundUpToMultiple(125, 32), 128);
    EXPECT_EQ(RoundUpToMultiple(128, 32), 128);
}

TEST(UtilsTest, IsPrime)
{
    EXPECT_EQ(IsPrime(1), false);
    EXPECT_EQ(IsPrime(2), true);
    EXPECT_EQ(IsPrime(3), true);
    EXPECT_EQ(IsPrime(4), false);
    EXPECT_EQ(IsPrime(5), true);
    EXPECT_EQ(IsPrime(6), false);
    EXPECT_EQ(IsPrime(7), true);
    EXPECT_EQ(IsPrime(8), false);
    EXPECT_EQ(IsPrime(9), false);
    EXPECT_EQ(IsPrime(10), false);
    EXPECT_EQ(IsPrime(11), true);
    EXPECT_EQ(IsPrime(12), false);
    EXPECT_EQ(IsPrime(13), true);
    EXPECT_EQ(IsPrime(14), false);
}

TEST(UtilsTest, NextPrime)
{
    EXPECT_EQ(NextPrime(1), 2);
    EXPECT_EQ(NextPrime(2), 2);
    EXPECT_EQ(NextPrime(3), 3);
    EXPECT_EQ(NextPrime(4), 5);
    EXPECT_EQ(NextPrime(5), 5);
    EXPECT_EQ(NextPrime(6), 7);
    EXPECT_EQ(NextPrime(7), 7);
    EXPECT_EQ(NextPrime(8), 11);
    EXPECT_EQ(NextPrime(9), 11);
    EXPECT_EQ(NextPrime(10), 11);
    EXPECT_EQ(NextPrime(11), 11);
    EXPECT_EQ(NextPrime(12), 13);
    EXPECT_EQ(NextPrime(13), 13);
    EXPECT_EQ(NextPrime(14), 17);

    EXPECT_EQ(NextPrime(200), 211);
}

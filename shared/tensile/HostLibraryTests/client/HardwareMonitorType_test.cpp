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

#include <HardwareMonitorType.hpp>

using namespace Tensile;
using namespace Tensile::Client;

// ClockType must stay a contiguous range bounded by CLK_TYPE_FIRST/CLK_TYPE_LAST
// so callers can iterate the clock types. These assertions catch reordering or
// renumbering of the live enumerators (they do not, and cannot at runtime,
// detect re-addition of an out-of-range sentinel such as the removed
// CLK_INVALID = 0xFFFFFFFF).
TEST(HardwareMonitorTypeTest, ClockTypeBounds)
{
    EXPECT_EQ(CLK_TYPE_FIRST, CLK_TYPE_SYS);
    EXPECT_EQ(CLK_TYPE_LAST, CLK_TYPE_MEM);
}

TEST(HardwareMonitorTypeTest, ClockTypeIsContiguous)
{
    EXPECT_EQ(CLK_TYPE_SYS, 0x0);
    EXPECT_EQ(CLK_TYPE_DF, CLK_TYPE_SYS + 1);
    EXPECT_EQ(CLK_TYPE_DCEF, CLK_TYPE_DF + 1);
    EXPECT_EQ(CLK_TYPE_SOC, CLK_TYPE_DCEF + 1);
    EXPECT_EQ(CLK_TYPE_MEM, CLK_TYPE_SOC + 1);
}

TEST(HardwareMonitorTypeTest, ClockTypeCountFromBounds)
{
    // FIRST..LAST inclusive must cover exactly the five live clock types.
    EXPECT_EQ(CLK_TYPE_LAST - CLK_TYPE_FIRST + 1, 5);
}

// Readings pass through unscaled; the rocm-smi path's /1000 would report 63 C as 0.063 C.
TEST(HardwareMonitorTypeTest, TempIsCelsiusNotMillidegrees)
{
    EXPECT_DOUBLE_EQ(smiTempSumToCelsius(63, 1), 63.0);
    EXPECT_DOUBLE_EQ(smiTempSumToCelsius(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(smiTempSumToCelsius(105, 1), 105.0);
    EXPECT_NE(smiTempSumToCelsius(63, 1), 63.0 / 1000.0);
}

// The accumulated sum is divided by the number of samples taken.
TEST(HardwareMonitorTypeTest, TempAveragesOverDataPoints)
{
    EXPECT_DOUBLE_EQ(smiTempSumToCelsius(60 + 62 + 64, 3), 62.0);
    EXPECT_DOUBLE_EQ(smiTempSumToCelsius(45 + 46, 2), 45.5);
}

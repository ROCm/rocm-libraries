/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>

#include <Rotating.hpp>

using TensileLite::clampRotatingNum;

// When the requested rotating copies comfortably fit in the allocated pool the
// count is returned unchanged.
TEST(RotatingBuffer, FitsUnchanged)
{
    // 10 copies of 100 bytes = 1000 bytes needed, 100000 available.
    EXPECT_EQ(clampRotatingNum(10, 100, 100000), 10);
}

// Reproducer from the bug report: a ~4 GiB rotating buffer where the requested
// count (42) over-provisions relative to the ~3.8 GiB actually allocated. The
// naive product 42 * 102235776 = 4293902592 also overflows a signed 32-bit int
// (it prints as -1064704), so the old code both over-counted and mis-compared.
// The clamp must return floor(4083829632 / 102235776) = 39.
TEST(RotatingBuffer, OverProvisionClampsToFit)
{
    EXPECT_EQ(clampRotatingNum(42, 102235776, 4083829632ULL), 39);
}

// When rotatingNum * rotatingSize exceeds INT32_MAX but the allocation is
// genuinely large enough, nothing is clamped. With 32-bit arithmetic the
// product would be negative and this would falsely clamp (or abort).
TEST(RotatingBuffer, NoOverflowWhenAllocationLargeEnough)
{
    // Exactly 42 copies allocated: 42 * 102235776 = 4293902592 (> INT32_MAX).
    EXPECT_EQ(clampRotatingNum(42, 102235776, 4293902592ULL), 42);
    // One extra byte of headroom: still no clamp.
    EXPECT_EQ(clampRotatingNum(42, 102235776, 4293902593ULL), 42);
    // One byte short: clamp down to 41.
    EXPECT_EQ(clampRotatingNum(42, 102235776, 4293902591ULL), 41);
}

// If not even a single copy fits, clamp to 0 (run with the original buffer only)
// instead of aborting.
TEST(RotatingBuffer, ZeroWhenNothingFits)
{
    EXPECT_EQ(clampRotatingNum(5, 102235776, 100), 0);
}

// Degenerate inputs are handled without dividing by zero or returning negatives.
TEST(RotatingBuffer, DegenerateInputs)
{
    EXPECT_EQ(clampRotatingNum(0, 100, 100000), 0);
    EXPECT_EQ(clampRotatingNum(-3, 100, 100000), 0);
    EXPECT_EQ(clampRotatingNum(5, 0, 100000), 5);
}

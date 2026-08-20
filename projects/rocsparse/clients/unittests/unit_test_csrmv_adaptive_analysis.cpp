/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

//
// Phase-2 unit tests: rocsparse::numThreadsForReduction() and rocsparse::flp2().
//
// These pure host helpers were lifted (behavior-preserving) out of the device
// translation unit
//   library/src/level2/rocsparse_csrmv_template_adaptive.cpp
// into the header
//   library/src/include/rocsparse_csrmv_adaptive_analysis.hpp
// so they can be tested directly without linking device code.
//
#include "rocsparse_csrmv_adaptive_analysis.hpp"

#include <cstdint>
#include <gtest/gtest.h>

// flp2(x) returns the largest power of two <= x (for x > 0).
TEST(csrmv_adaptive_analysis, flp2_basic)
{
    EXPECT_EQ(rocsparse::flp2(1u), 1u);
    EXPECT_EQ(rocsparse::flp2(2u), 2u);
    EXPECT_EQ(rocsparse::flp2(3u), 2u);
    EXPECT_EQ(rocsparse::flp2(4u), 4u);
    EXPECT_EQ(rocsparse::flp2(5u), 4u);
    EXPECT_EQ(rocsparse::flp2(7u), 4u);
    EXPECT_EQ(rocsparse::flp2(8u), 8u);
    EXPECT_EQ(rocsparse::flp2(255u), 128u);
    EXPECT_EQ(rocsparse::flp2(256u), 256u);
    EXPECT_EQ(rocsparse::flp2(1000u), 512u);
}

// Exact powers of two are fixed points of flp2.
TEST(csrmv_adaptive_analysis, flp2_powers_of_two_are_fixed_points)
{
    for(uint32_t p = 0; p < 31; ++p)
    {
        const uint32_t pow2 = (1u << p);
        EXPECT_EQ(rocsparse::flp2(pow2), pow2) << "p=" << p;
    }
}

// numThreadsForReduction(num_rows) with the default workgroup size (256).
// result == 256 >> bit_width(num_rows - 1). Values below are hand-computed and
// match the algorithm comment (4 rows -> 64 threads, 5 rows -> 32 threads).
TEST(csrmv_adaptive_analysis, numThreadsForReduction_default_wg)
{
    EXPECT_EQ(rocsparse::numThreadsForReduction(2), 128u);
    EXPECT_EQ(rocsparse::numThreadsForReduction(3), 64u);
    EXPECT_EQ(rocsparse::numThreadsForReduction(4), 64u);
    EXPECT_EQ(rocsparse::numThreadsForReduction(5), 32u);
    EXPECT_EQ(rocsparse::numThreadsForReduction(8), 32u);
    EXPECT_EQ(rocsparse::numThreadsForReduction(9), 16u);
    EXPECT_EQ(rocsparse::numThreadsForReduction(16), 16u);
    EXPECT_EQ(rocsparse::numThreadsForReduction(17), 8u);
}

// Boundary behavior: once (num_rows - 1) needs 8 bits the result collapses to
// a single thread, and beyond the workgroup size it saturates to zero.
TEST(csrmv_adaptive_analysis, numThreadsForReduction_boundaries)
{
    EXPECT_EQ(rocsparse::numThreadsForReduction(129), 1u); // 128 -> 8 bits -> 256>>8
    EXPECT_EQ(rocsparse::numThreadsForReduction(256), 1u); // 255 -> 8 bits -> 256>>8
    EXPECT_EQ(rocsparse::numThreadsForReduction(257), 0u); // 256 -> 9 bits -> 256>>9
    EXPECT_EQ(rocsparse::numThreadsForReduction(1000), 0u);
}

// Result is monotonically non-increasing as num_rows grows (more rows batched
// together => fewer threads can team up per row).
TEST(csrmv_adaptive_analysis, numThreadsForReduction_monotonic_non_increasing)
{
    uint64_t prev = rocsparse::numThreadsForReduction(2);
    for(uint64_t num_rows = 3; num_rows <= 512; ++num_rows)
    {
        const uint64_t cur = rocsparse::numThreadsForReduction(num_rows);
        EXPECT_LE(cur, prev) << "num_rows=" << num_rows;
        prev = cur;
    }
}

// The workgroup size is a parameter (defaulting to 256); the result scales
// with it for the same num_rows.
TEST(csrmv_adaptive_analysis, numThreadsForReduction_custom_wg_size)
{
    EXPECT_EQ(rocsparse::numThreadsForReduction(4, 128), 32u); // 128 >> bit_width(3)=2
    EXPECT_EQ(rocsparse::numThreadsForReduction(4, 512), 128u); // 512 >> 2
    EXPECT_EQ(rocsparse::numThreadsForReduction(5, 64), 8u); // 64 >> bit_width(4)=3
}

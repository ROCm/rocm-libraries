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
// The medium path (m > 1024) of gtsv_no_pivot and gtsv_no_pivot_strided_batch
// launches straight-line kernels with 32-bit indices when the launch fits in
// grid.y and those indices cannot overflow, and grid-stride kernels with 64-bit
// indices otherwise. The 32-bit limits are only reached by problems far larger
// than any test can allocate, so the selection is checked here on the host at
// each limit and one past it.
//
#include "../../library/src/precond/gtsv/gtsv_nopivot_index.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <limits>

namespace
{
    constexpr int64_t k_uint32_max = std::numeric_limits<uint32_t>::max();
    constexpr int64_t k_int32_max  = std::numeric_limits<int32_t>::max();
    constexpr int64_t k_int64_max  = std::numeric_limits<int64_t>::max();

    // Block size of the medium kernels: num_spikes = 2 * ceil(m / 256).
    int64_t num_spikes_of(int64_t m)
    {
        return 2 * ((m - 1) / 256 + 1);
    }
}

TEST(GtsvNoPivotIndex, index_fits_limit)
{
    // stride * columns + inner == limit fits, limit + 1 does not.
    EXPECT_TRUE(rocsparse::gtsv_index_fits(1000, 7, k_uint32_max - 7000, k_uint32_max));
    EXPECT_FALSE(rocsparse::gtsv_index_fits(1000, 7, k_uint32_max - 6999, k_uint32_max));
    EXPECT_TRUE(rocsparse::gtsv_index_fits(1000, 7, k_int32_max - 7000, k_int32_max));
    EXPECT_FALSE(rocsparse::gtsv_index_fits(1000, 7, k_int32_max - 6999, k_int32_max));
    EXPECT_TRUE(rocsparse::gtsv_index_fits(65536, 65535, 65535, k_uint32_max));
    EXPECT_FALSE(rocsparse::gtsv_index_fits(65536, 65536, 0, k_uint32_max));
}

TEST(GtsvNoPivotIndex, index_fits_zero_columns)
{
    // columns == 0 checks inner alone, whatever the stride.
    EXPECT_TRUE(rocsparse::gtsv_index_fits(k_int64_max, 0, k_uint32_max, k_uint32_max));
    EXPECT_FALSE(rocsparse::gtsv_index_fits(k_int64_max, 0, k_uint32_max + 1, k_uint32_max));
    EXPECT_TRUE(rocsparse::gtsv_index_fits(0, 0, 0, 0));
}

TEST(GtsvNoPivotIndex, index_fits_int64_overflow)
{
    // stride * columns would overflow int64_t: rejected rather than wrapped.
    EXPECT_FALSE(rocsparse::gtsv_index_fits(int64_t{1} << 32, int64_t{1} << 32, 0, k_int64_max));
    EXPECT_FALSE(rocsparse::gtsv_index_fits(k_int64_max, 2, 0, k_int64_max));
    EXPECT_TRUE(rocsparse::gtsv_index_fits(k_int64_max / 2, 2, 1, k_int64_max));
    EXPECT_FALSE(rocsparse::gtsv_index_fits(k_int64_max / 2, 2, 2, k_int64_max));
}

TEST(GtsvNoPivotIndex, index_fits_negative)
{
    EXPECT_FALSE(rocsparse::gtsv_index_fits(-1, 1, 0, k_uint32_max));
    EXPECT_FALSE(rocsparse::gtsv_index_fits(1, -1, 0, k_uint32_max));
    EXPECT_FALSE(rocsparse::gtsv_index_fits(1, 1, -1, k_uint32_max));
}

TEST(GtsvNoPivotIndex, medium_straight_grid_y)
{
    // n right-hand sides in tiles of 8 need ceil(n / 8) blocks in grid.y.
    const int64_t m = 1025;
    const int64_t s = num_spikes_of(m);
    EXPECT_TRUE(rocsparse::gtsv_no_pivot_medium_straight<int32_t>(m, 1, 8, s, 1));
    EXPECT_TRUE(rocsparse::gtsv_no_pivot_medium_straight<int32_t>(m, 8, 8, s, 1));
    EXPECT_FALSE(rocsparse::gtsv_no_pivot_medium_straight<int32_t>(m, 9, 8, s, 1));
    EXPECT_TRUE(rocsparse::gtsv_no_pivot_medium_straight<int32_t>(m, 24, 8, s, 3));
    EXPECT_FALSE(rocsparse::gtsv_no_pivot_medium_straight<int32_t>(m, 25, 8, s, 3));
    EXPECT_TRUE(rocsparse::gtsv_no_pivot_medium_straight<int32_t>(m, 524280, 8, s, 65535));
    EXPECT_FALSE(rocsparse::gtsv_no_pivot_medium_straight<int32_t>(m, 524281, 8, s, 65535));
    EXPECT_TRUE(rocsparse::gtsv_no_pivot_medium_straight<int64_t>(m, 524280, 8, s, 65535));
    EXPECT_FALSE(rocsparse::gtsv_no_pivot_medium_straight<int64_t>(m, 524281, 8, s, 65535));
}

TEST(GtsvNoPivotIndex, medium_straight_modified_limit)
{
    // m * (n - 1) + m - 1 = m * n - 1 must fit in 32 bits unsigned for a 32-bit
    // rocsparse_int. m = 2^16 + 1 with n = 65535 gives m * n - 1 = 2^32 - 2.
    const int64_t m = 65537;
    const int64_t s = num_spikes_of(m);
    ASSERT_EQ(m * 65535 - 1, k_uint32_max - 1);
    EXPECT_TRUE(rocsparse::gtsv_no_pivot_medium_straight<int32_t>(m, 65535, 8, s, 1 << 30));
    // m = 2^16 with n = 2^16 gives m * n - 1 = 2^32 - 1, the limit.
    EXPECT_TRUE(rocsparse::gtsv_no_pivot_medium_straight<int32_t>(
        65536, 65536, 8, num_spikes_of(65536), 1 << 30));
    // One more right-hand side is one past it.
    EXPECT_FALSE(rocsparse::gtsv_no_pivot_medium_straight<int32_t>(
        65536, 65537, 8, num_spikes_of(65536), 1 << 30));
    // The AISPARSE-689 stress case: 8 * m - 1 is between 2^31 and 2^32.
    EXPECT_TRUE(rocsparse::gtsv_no_pivot_medium_straight<int32_t>(
        306783379, 8, 8, num_spikes_of(306783379), 65535));
    // A 64-bit rocsparse_int forms the temporary buffer index in 64 bit.
    EXPECT_TRUE(rocsparse::gtsv_no_pivot_medium_straight<int64_t>(
        65536, 65537, 8, num_spikes_of(65536), 1 << 30));
}

TEST(GtsvNoPivotIndex, medium_straight_spike_limit)
{
    // num_spikes * n - 1 must fit in 32 bits unsigned for both index widths. With
    // m = 2^23 there are 2^16 spikes, so the spike limit binds at n = 2^16 while
    // m * n - 1 = 2^39 - 1 only binds for a 32-bit rocsparse_int.
    const int64_t m = int64_t{1} << 23;
    const int64_t s = num_spikes_of(m);
    ASSERT_EQ(s, 65536);
    EXPECT_TRUE(rocsparse::gtsv_no_pivot_medium_straight<int64_t>(m, 65536, 8, s, 1 << 30));
    EXPECT_FALSE(rocsparse::gtsv_no_pivot_medium_straight<int64_t>(m, 65537, 8, s, 1 << 30));
    EXPECT_FALSE(rocsparse::gtsv_no_pivot_medium_straight<int32_t>(m, 65536, 8, s, 1 << 30));
}

TEST(GtsvNoPivotIndex, strided_medium_straight_grid_y)
{
    const int64_t m = 1025;
    const int64_t s = num_spikes_of(m);
    EXPECT_TRUE(rocsparse::gtsv_no_pivot_strided_medium_straight<int32_t>(m, 1, s, 1));
    EXPECT_FALSE(rocsparse::gtsv_no_pivot_strided_medium_straight<int32_t>(m, 2, s, 1));
    EXPECT_TRUE(rocsparse::gtsv_no_pivot_strided_medium_straight<int32_t>(m, 3, s, 3));
    EXPECT_FALSE(rocsparse::gtsv_no_pivot_strided_medium_straight<int32_t>(m, 4, s, 3));
    EXPECT_TRUE(rocsparse::gtsv_no_pivot_strided_medium_straight<int32_t>(m, 65535, s, 65535));
    EXPECT_FALSE(rocsparse::gtsv_no_pivot_strided_medium_straight<int32_t>(m, 65536, s, 65535));
}

TEST(GtsvNoPivotIndex, strided_medium_straight_rocsparse_int_limit)
{
    // m * batch_count - 1 must fit in a signed rocsparse_int. m = 32768 with 65536
    // systems gives 2^31 - 1, the 32-bit limit; one more system is one past it.
    const int64_t m = 32768;
    const int64_t s = num_spikes_of(m);
    EXPECT_TRUE(rocsparse::gtsv_no_pivot_strided_medium_straight<int32_t>(m, 65536, s, 1 << 30));
    EXPECT_FALSE(rocsparse::gtsv_no_pivot_strided_medium_straight<int32_t>(m, 65537, s, 1 << 30));
    EXPECT_TRUE(rocsparse::gtsv_no_pivot_strided_medium_straight<int64_t>(m, 65537, s, 1 << 30));
    // A long system: m = 2^23 reaches the limit at 256 systems.
    const int64_t m_long = int64_t{1} << 23;
    const int64_t s_long = num_spikes_of(m_long);
    EXPECT_TRUE(
        rocsparse::gtsv_no_pivot_strided_medium_straight<int32_t>(m_long, 256, s_long, 1 << 30));
    EXPECT_FALSE(
        rocsparse::gtsv_no_pivot_strided_medium_straight<int32_t>(m_long, 257, s_long, 1 << 30));
    EXPECT_TRUE(
        rocsparse::gtsv_no_pivot_strided_medium_straight<int64_t>(m_long, 32768, s_long, 1 << 30));
}

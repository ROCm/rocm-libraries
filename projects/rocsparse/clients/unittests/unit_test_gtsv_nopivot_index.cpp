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
// runs its forward and backward kernels over chunks of right-hand sides or
// systems, each small enough for grid.y and for the 32-bit indices of the
// kernels. The 32-bit limits are only reached by problems far larger than any
// test can allocate, so the chunk sizes are checked here on the host at each
// limit.
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

TEST(GtsvNoPivotIndex, max_count)
{
    // stride * count - 1 fits in limit, stride * (count + 1) - 1 does not.
    EXPECT_EQ(rocsparse::gtsv_max_count(1000, k_uint32_max), 4294967);
    EXPECT_LE(1000 * int64_t{4294967} - 1, k_uint32_max);
    EXPECT_GT(1000 * int64_t{4294968} - 1, k_uint32_max);
    EXPECT_EQ(rocsparse::gtsv_max_count(65536, k_uint32_max), 65536);
    EXPECT_EQ(rocsparse::gtsv_max_count(1025, k_int32_max), 2095105);
    EXPECT_LE(1025 * int64_t{2095105} - 1, k_int32_max);
    EXPECT_GT(1025 * int64_t{2095106} - 1, k_int32_max);
    EXPECT_EQ(rocsparse::gtsv_max_count(k_uint32_max, k_uint32_max), 1);
    // A 64-bit limit does not overflow.
    EXPECT_EQ(rocsparse::gtsv_max_count(k_int64_max / 2, k_int64_max), 2);
    EXPECT_EQ(rocsparse::gtsv_max_count(k_int64_max, k_int64_max), 1);
}

TEST(GtsvNoPivotIndex, medium_chunk_grid_y)
{
    // One block per tile of 8 right-hand sides in grid.y.
    const int64_t m = 1025;
    const int64_t s = num_spikes_of(m);
    EXPECT_EQ(rocsparse::gtsv_no_pivot_medium_chunk<int32_t>(m, 8, s, 1), 8);
    EXPECT_EQ(rocsparse::gtsv_no_pivot_medium_chunk<int32_t>(m, 8, s, 3), 24);
    EXPECT_EQ(rocsparse::gtsv_no_pivot_medium_chunk<int32_t>(m, 8, s, 65535), 524280);
    EXPECT_EQ(rocsparse::gtsv_no_pivot_medium_chunk<int64_t>(m, 8, s, 65535), 524280);
}

TEST(GtsvNoPivotIndex, medium_chunk_modified_limit)
{
    // m * columns - 1 must fit in 32 bits unsigned for a 32-bit rocsparse_int.
    // m = 2^16 gives 2^16 columns, a multiple of the tile.
    EXPECT_EQ(
        rocsparse::gtsv_no_pivot_medium_chunk<int32_t>(65536, 8, num_spikes_of(65536), 1 << 30),
        65536);
    // m = 2^16 + 1 gives 65535 columns, rounded down to whole tiles.
    EXPECT_EQ(
        rocsparse::gtsv_no_pivot_medium_chunk<int32_t>(65537, 8, num_spikes_of(65537), 1 << 30),
        65528);
    // The AISPARSE-689 stress case: 8 * m - 1 is between 2^31 and 2^32, so its
    // n = 8 right-hand sides take a single launch.
    EXPECT_EQ(rocsparse::gtsv_no_pivot_medium_chunk<int32_t>(
                  306783379, 8, num_spikes_of(306783379), 65535),
              8);
    // Past m = 2^29 not even one tile fits: the chunk is the columns that do.
    EXPECT_EQ(rocsparse::gtsv_no_pivot_medium_chunk<int32_t>(
                  int64_t{1} << 30, 8, num_spikes_of(int64_t{1} << 30), 65535),
              4);
    EXPECT_EQ(rocsparse::gtsv_no_pivot_medium_chunk<int32_t>(
                  k_int32_max, 8, num_spikes_of(k_int32_max), 65535),
              2);
    // A 64-bit rocsparse_int forms the temporary buffer index in 64 bit, and only
    // the spike index limits the chunk: 2^32 / 512 spikes.
    EXPECT_EQ(
        rocsparse::gtsv_no_pivot_medium_chunk<int64_t>(65536, 8, num_spikes_of(65536), 1 << 30),
        8388608);
}

TEST(GtsvNoPivotIndex, medium_chunk_spike_limit)
{
    // num_spikes * columns - 1 must fit in 32 bits unsigned for both index widths.
    // With m = 2^23 there are 2^16 spikes, so the spike limit binds at 2^16
    // columns, while m * columns - 1 binds first at 512 for a 32-bit rocsparse_int.
    const int64_t m = int64_t{1} << 23;
    const int64_t s = num_spikes_of(m);
    ASSERT_EQ(s, 65536);
    EXPECT_EQ(rocsparse::gtsv_no_pivot_medium_chunk<int64_t>(m, 8, s, 1 << 30), 65536);
    EXPECT_EQ(rocsparse::gtsv_no_pivot_medium_chunk<int32_t>(m, 8, s, 1 << 30), 512);
}

TEST(GtsvNoPivotIndex, strided_medium_chunk_grid_y)
{
    // One block per system in grid.y.
    const int64_t m = 1025;
    const int64_t s = num_spikes_of(m);
    EXPECT_EQ(rocsparse::gtsv_no_pivot_strided_medium_chunk<int32_t>(m, s, 1), 1);
    EXPECT_EQ(rocsparse::gtsv_no_pivot_strided_medium_chunk<int32_t>(m, s, 3), 3);
    EXPECT_EQ(rocsparse::gtsv_no_pivot_strided_medium_chunk<int32_t>(m, s, 65535), 65535);
}

TEST(GtsvNoPivotIndex, strided_medium_chunk_rocsparse_int_limit)
{
    // m * count - 1 must fit in a signed rocsparse_int: m = 32768 gives 65536
    // systems for 32 bits.
    const int64_t m = 32768;
    const int64_t s = num_spikes_of(m);
    EXPECT_EQ(rocsparse::gtsv_no_pivot_strided_medium_chunk<int32_t>(m, s, 1 << 30), 65536);
    EXPECT_EQ(rocsparse::gtsv_no_pivot_strided_medium_chunk<int64_t>(m, s, 1 << 30), 1 << 30);
    // A long system: m = 2^23 gives 256 systems.
    const int64_t m_long = int64_t{1} << 23;
    EXPECT_EQ(rocsparse::gtsv_no_pivot_strided_medium_chunk<int32_t>(
                  m_long, num_spikes_of(m_long), 1 << 30),
              256);
    // The largest system still takes one per launch.
    EXPECT_EQ(rocsparse::gtsv_no_pivot_strided_medium_chunk<int32_t>(
                  k_int32_max, num_spikes_of(k_int32_max), 1 << 30),
              1);
}

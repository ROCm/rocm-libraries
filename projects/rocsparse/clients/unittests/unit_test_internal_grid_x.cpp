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
// Host unit tests for the grid.x clamp and for the contiguous chunk partition
// that the incomplete factorizations use to recover the work above the clamp.
//
// These live in the host-only rocsparse-unit-test binary rather than in
// rocsparse-unit-test-device, and they declare no memory requirement, so
// nothing can drop them at instantiation time. The whole row count range the
// clamp exists for is far past what any available device can hold, so the
// properties the kernels depend on are checked here arithmetically instead:
//
//   - the clamp never returns an extent the hardware would reject, and never
//     truncates an extent it would accept;
//   - the chunks are a partition: every item in [0, count) is owned by exactly
//     one block;
//   - the chunks are monotone in the block id, which is the property that keeps
//     the done-flag spin in the factorization kernels from deadlocking;
//   - at any unclamped extent the partition degenerates to one item per block,
//     so the recovery loop is a no-op for every reachable problem size.
//
#include "rocsparse_grid_x.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

TEST(grid_x, max_extent_matches_the_hardware)
{
    // gfx1201 reports maxGridSize[0] == 2147483647. Every AMD GPU reports the
    // same value for the x axis; the 65535 cap applies to y and z only.
    EXPECT_EQ(rocsparse::grid_x_max_extent(), 2147483647);
}

TEST(grid_x, get_grid_size_x_below_and_at_the_clamp)
{
    EXPECT_EQ(rocsparse::get_grid_size_x(0), 0u);
    EXPECT_EQ(rocsparse::get_grid_size_x(1), 1u);
    EXPECT_EQ(rocsparse::get_grid_size_x(65536), 65536u);
    EXPECT_EQ(rocsparse::get_grid_size_x(2147483646), 2147483646u);
    EXPECT_EQ(rocsparse::get_grid_size_x(2147483647), 2147483647u);
}

TEST(grid_x, get_grid_size_x_above_the_clamp)
{
    // Without the clamp these narrow into dim3: 2^31 and 2^31 + 1 are accepted
    // by dim3 but rejected by the driver, and 2^32 and 2^32 + 7 wrap to 0 and 7,
    // so the launch would silently do nothing or almost nothing.
    EXPECT_EQ(rocsparse::get_grid_size_x(2147483648LL), 2147483647u);
    EXPECT_EQ(rocsparse::get_grid_size_x(2147483649LL), 2147483647u);
    EXPECT_EQ(rocsparse::get_grid_size_x(4294967296LL), 2147483647u);
    EXPECT_EQ(rocsparse::get_grid_size_x(4294967303LL), 2147483647u);
    EXPECT_EQ(rocsparse::get_grid_size_x(1LL << 40), 2147483647u);
}

TEST(grid_x, grid_x_chunk_is_a_partition)
{
    for(int64_t count : {int64_t(1),
                         int64_t(2),
                         int64_t(5),
                         int64_t(16),
                         int64_t(97),
                         int64_t(1024),
                         int64_t(4097)})
    {
        for(uint32_t grid_size = 1; grid_size <= static_cast<uint32_t>(count); ++grid_size)
        {
            std::vector<int> owners(static_cast<size_t>(count), 0);

            for(uint32_t block_id = 0; block_id < grid_size; ++block_id)
            {
                int64_t first, last;
                rocsparse::grid_x_chunk(count, grid_size, block_id, first, last);

                ASSERT_LE(first, last) << "count " << count << " grid_size " << grid_size;

                for(int64_t i = first; i < last; ++i)
                {
                    ASSERT_GE(i, 0);
                    ASSERT_LT(i, count);
                    ++owners[static_cast<size_t>(i)];
                }
            }

            for(int64_t i = 0; i < count; ++i)
            {
                ASSERT_EQ(owners[static_cast<size_t>(i)], 1)
                    << "item " << i << " of " << count << " with grid_size " << grid_size;
            }
        }
    }
}

TEST(grid_x, grid_x_chunk_is_monotone_in_the_block_id)
{
    // A block only ever waits on items belonging to itself or to a lower
    // numbered block. A grid-stride loop would not have this property and would
    // deadlock the done-flag spin once the grid exceeds what stays resident.
    for(int64_t count : {int64_t(7), int64_t(100), int64_t(4097)})
    {
        for(uint32_t grid_size = 1; grid_size <= static_cast<uint32_t>(count); ++grid_size)
        {
            int64_t previous_last = 0;

            for(uint32_t block_id = 0; block_id < grid_size; ++block_id)
            {
                int64_t first, last;
                rocsparse::grid_x_chunk(count, grid_size, block_id, first, last);

                if(first < last)
                {
                    ASSERT_GE(first, previous_last)
                        << "count " << count << " grid_size " << grid_size << " block " << block_id;
                    previous_last = last;
                }
            }

            ASSERT_EQ(previous_last, count);
        }
    }
}

TEST(grid_x, grid_x_chunk_degenerates_to_one_item_per_block_when_unclamped)
{
    // Every problem size that fits on hardware today launches grid_size ==
    // count, and this is what makes the recovery loop a no-op there.
    for(int64_t count : {int64_t(1), int64_t(3), int64_t(64), int64_t(65537), int64_t(100000)})
    {
        const uint32_t grid_size = rocsparse::get_grid_size_x(count);
        ASSERT_EQ(static_cast<int64_t>(grid_size), count);

        for(uint32_t block_id = 0; block_id < grid_size; ++block_id)
        {
            int64_t first, last;
            rocsparse::grid_x_chunk(count, grid_size, block_id, first, last);

            ASSERT_EQ(first, static_cast<int64_t>(block_id));
            ASSERT_EQ(last, static_cast<int64_t>(block_id) + 1);
        }
    }
}

TEST(grid_x, grid_x_chunk_covers_a_count_above_the_clamp)
{
    // The case the clamp exists for: more items than the grid can express.
    // Check the leading blocks and the final block without walking 2^31 items.
    const int64_t  count     = 5000000000LL;
    const uint32_t grid_size = rocsparse::get_grid_size_x(count);

    ASSERT_EQ(grid_size, 2147483647u);

    int64_t first, last;

    rocsparse::grid_x_chunk(count, grid_size, 0, first, last);
    EXPECT_EQ(first, 0);
    EXPECT_EQ(last, 3);

    int64_t total = 0;
    for(uint32_t block_id = 0; block_id < 1000; ++block_id)
    {
        rocsparse::grid_x_chunk(count, grid_size, block_id, first, last);
        total += last - first;
    }
    EXPECT_EQ(total, 3000);

    rocsparse::grid_x_chunk(count, grid_size, grid_size - 1, first, last);
    EXPECT_EQ(last, count);
    EXPECT_LE(first, last);
}

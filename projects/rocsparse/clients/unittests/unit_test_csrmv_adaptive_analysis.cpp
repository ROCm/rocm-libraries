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
//   library/src/level2/rocsparse_csrmv_adaptive_analysis.hpp
// so they can be tested directly without linking device code.
//
#include "../../library/src/level2/rocsparse_csrmv_adaptive_analysis.hpp"
#include "../../library/src/level2/rocsparse_csrmv_partition.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

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

namespace
{
    bool poor_fit(const std::vector<int>& blocks, const std::vector<int>& ptr)
    {
        rocsparse::csrmv_partition<int, int> partition;
        partition.row_blocks = blocks;
        partition.row_ptr    = ptr;
        return partition.is_poor_fit();
    }
}

// csrmv_partition::is_poor_fit decides when default SpMV selects nnz-split.
// A block is wide when it spans more than csrmv_adaptive_constants::wide_block_rows rows.
TEST(csrmv_adaptive_analysis, poor_fit_wide_blocks_covering_the_matrix)
{
    const int        n = 10000;
    std::vector<int> blocks{0, n};
    std::vector<int> ptr(static_cast<size_t>(n) + 1, 1);
    ptr.front() = 0;
    ptr.back()  = 2;
    EXPECT_TRUE(poor_fit(blocks, ptr));
}

// The limit is fixed: general_wg_size returns 128 for some matrices, and
// 32 * 128 rows must not be the limit.
TEST(csrmv_adaptive_analysis, poor_fit_span_limit_is_fixed)
{
    auto gap = [](int span) {
        const int        populated = 1000;
        const int        n         = span + populated;
        std::vector<int> blocks{0, span, n};
        std::vector<int> ptr{0};
        for(int row = 0; row < n; ++row)
        {
            const bool empty = row > 0 && row < span - 1;
            ptr.push_back(ptr.back() + (empty ? 0 : 1));
        }
        return poor_fit(blocks, ptr);
    };
    EXPECT_FALSE(gap(4097));
    EXPECT_FALSE(gap(8192));
    EXPECT_TRUE(gap(8193));
}

// Adaptive does not close a block on empty rows, so the first block holds the
// empty prefix and the populated rows up to a full block of nonzeros. A prefix
// longer than the limit is trimmed by adaptive; a shorter one stays in the
// block. Neither counts towards the span.
TEST(csrmv_adaptive_analysis, good_fit_with_an_empty_prefix)
{
    auto prefix = [](int edge) {
        const int        populated = 4 * 1024;
        const int        n         = edge + populated;
        const int        first     = edge > 8192 ? edge - 2 : 0;
        std::vector<int> blocks{first};
        for(int b = 1; b <= 4; ++b)
        {
            blocks.push_back(edge + b * 1024);
        }
        std::vector<int> ptr(static_cast<size_t>(edge) + 1, 0);
        for(int row = 0; row < populated; ++row)
        {
            ptr.push_back(row + 1);
        }
        return poor_fit(blocks, ptr);
    };
    EXPECT_FALSE(prefix(4097));
    EXPECT_FALSE(prefix(8000));
    EXPECT_FALSE(prefix(8192));
    EXPECT_FALSE(prefix(20000));
}

// After trimming a long empty suffix, the last block can hold only the few
// empty rows adaptive keeps. It is an edge, not an underfilled block.
TEST(csrmv_adaptive_analysis, trimmed_empty_suffix_is_not_underfilled)
{
    const int        suffix = 9000;
    const int        n      = 9 + suffix;
    std::vector<int> blocks{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12};
    std::vector<int> ptr{0};
    for(int row = 0; row < n; ++row)
    {
        const int len = row < 7 ? 2000 : (row < 9 ? 10 : 0);
        ptr.push_back(ptr.back() + len);
    }
    EXPECT_FALSE(poor_fit(blocks, ptr));
}

// A wide block that is only a small share of the matrix, and no transition
// fragments, is a good fit.
TEST(csrmv_adaptive_analysis, good_fit_when_wide_blocks_are_a_small_share)
{
    const int        wide = 8193;
    const int        rest = 9;
    const int        span = 8192;
    const int        n    = wide + rest * span;
    std::vector<int> blocks{0};
    std::vector<int> ptr{0};
    int              row  = 0;
    int              nnz  = 0;
    auto             push = [&](int rows, int row_nnz) {
        blocks.push_back(row + rows);
        for(int r = 0; r < rows; ++r)
        {
            nnz += row_nnz;
            ptr.push_back(nnz);
        }
        row += rows;
    };
    push(wide, 200);
    for(int b = 0; b < rest; ++b)
    {
        push(span, 200);
    }
    EXPECT_FALSE(poor_fit(blocks, ptr));
}

TEST(csrmv_adaptive_analysis, poor_fit_when_a_quarter_of_blocks_are_underfilled)
{
    std::vector<int> blocks{0};
    std::vector<int> ptr{0};
    int              nnz = 0;
    for(int row = 0; row < 8; ++row)
    {
        nnz += (row % 2 == 0) ? 10 : 200;
        ptr.push_back(nnz);
        blocks.push_back(row + 1);
    }
    EXPECT_TRUE(poor_fit(blocks, ptr));
}

TEST(csrmv_adaptive_analysis, good_fit_below_a_quarter_underfilled)
{
    std::vector<int> blocks{0};
    std::vector<int> ptr{0};
    int              nnz = 0;
    for(int row = 0; row < 8; ++row)
    {
        nnz += (row == 0) ? 10 : 200;
        ptr.push_back(nnz);
        blocks.push_back(row + 1);
    }
    EXPECT_FALSE(poor_fit(blocks, ptr));
}

// Seven underfilled blocks are not enough to call the pattern systematic.
TEST(csrmv_adaptive_analysis, good_fit_without_enough_blocks)
{
    std::vector<int> blocks{0};
    std::vector<int> ptr{0};
    int              nnz = 0;
    for(int row = 0; row < 7; ++row)
    {
        nnz += 10;
        ptr.push_back(nnz);
        blocks.push_back(row + 1);
    }
    EXPECT_FALSE(poor_fit(blocks, ptr));
}

// Intermediate chunks of one long row have a zero row span and are ignored.
// Counted, the zero-span entry would be an eighth comparable block, and an
// underfilled one.
TEST(csrmv_adaptive_analysis, zero_span_chunks_are_ignored)
{
    const std::vector<int> blocks{0, 1, 2, 3, 4, 5, 6, 7, 7};
    std::vector<int>       ptr;
    for(int row = 0; row <= 7; ++row)
    {
        ptr.push_back(10 * row);
    }
    EXPECT_FALSE(poor_fit(blocks, ptr));
}

// Empty rows between long rows become one-row blocks without nonzeros. They do
// as little work as one-nonzero blocks and count as underfilled the same way.
TEST(csrmv_adaptive_analysis, poor_fit_counts_empty_blocks_as_underfilled)
{
    for(const int short_nnz : {0, 1})
    {
        std::vector<int> blocks{0};
        std::vector<int> ptr{0};
        int              nnz = 0;
        for(int row = 0; row < 16; ++row)
        {
            nnz += (row % 2 == 0) ? 2000 : short_nnz;
            ptr.push_back(nnz);
            blocks.push_back(row + 1);
        }
        EXPECT_TRUE(poor_fit(blocks, ptr)) << "short_nnz=" << short_nnz;
    }
}

TEST(csrmv_adaptive_analysis, good_fit_for_empty_partition)
{
    const std::vector<int> blocks{0};
    const std::vector<int> ptr{0};
    EXPECT_FALSE(poor_fit(blocks, ptr));
}

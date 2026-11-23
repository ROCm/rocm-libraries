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
// Tier-0 unit tests: rocsparse::bsr_block_count_matches<I>(m, mb, block_dim) and
// rocsparse::bsr_diag_offset<I>(block_index, i, block_dim).
//
// Header-only helpers from library/src/include/rocsparse_bsr_dims.hpp. The first
// validates the BSR block-row count, (mb - 1) * block_dim < m <= mb * block_dim.
// It must hold for every representable input of both index widths, including
// shapes whose products do not fit in I (GitHub issue #15913). The second is the
// bsrpad_value write offset, which must not wrap at 32 bits.
//
#include "rocsparse_bsr_dims.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <limits>

namespace
{
    // Exact in int64_t while |m|, |mb|, |block_dim| <= INT32_MAX.
    bool shape_relation(int64_t m, int64_t mb, int64_t block_dim)
    {
        return (mb - 1) * block_dim < m && m <= mb * block_dim;
    }

    template <typename I>
    void check_small_exhaustive()
    {
        for(I block_dim = 1; block_dim < 64; ++block_dim)
        {
            for(I mb = 0; mb < 64; ++mb)
            {
                for(I m = 0; m < 4096; ++m)
                {
                    ASSERT_EQ(rocsparse::bsr_block_count_matches<I>(m, mb, block_dim),
                              shape_relation(m, mb, block_dim))
                        << "m=" << m << " mb=" << mb << " block_dim=" << block_dim;
                }
            }
        }
    }
}

TEST(bsr_dims, small_exhaustive_i32)
{
    check_small_exhaustive<int32_t>();
}

TEST(bsr_dims, small_exhaustive_i64)
{
    check_small_exhaustive<int64_t>();
}

TEST(bsr_dims, int32_limits)
{
    struct row
    {
        int32_t m, mb, block_dim;
    };
    constexpr int32_t imax   = std::numeric_limits<int32_t>::max();
    const row         rows[] = {
        {1, 3, 1431655766}, // #15913
        {4, 4, 1073741825},
        {1, imax, imax},
        {3, 1073741825, 4},
        {imax, 2, 1073741824},
        {1073741825, 2, 1073741824},
        {1073741824, 2, 1073741824},
        {imax, 3, 1073741823},
        {imax, 3, 1073741824},
        {imax - 1, 2, 1073741823},
        {1, 1, imax},
        {imax, 1, imax},
        {1, 2, imax},
        {0, 0, imax},
        {0, 1, 5},
        {1, 0, 1},
    };
    for(const auto& r : rows)
    {
        const bool expected = shape_relation(r.m, r.mb, r.block_dim);
        EXPECT_EQ(rocsparse::bsr_block_count_matches<int32_t>(r.m, r.mb, r.block_dim), expected)
            << "m=" << r.m << " mb=" << r.mb << " block_dim=" << r.block_dim;
        EXPECT_EQ(rocsparse::bsr_block_count_matches<int64_t>(r.m, r.mb, r.block_dim), expected)
            << "m=" << r.m << " mb=" << r.mb << " block_dim=" << r.block_dim;
    }
}

TEST(bsr_dims, int64_limits)
{
    constexpr int64_t lmax = std::numeric_limits<int64_t>::max();

    // 3 * 6148914691236517206 = 2^64 + 2.
    EXPECT_FALSE(rocsparse::bsr_block_count_matches<int64_t>(1, 3, 6148914691236517206));
    EXPECT_TRUE(rocsparse::bsr_block_count_matches<int64_t>(lmax, 1, lmax));
    EXPECT_TRUE(rocsparse::bsr_block_count_matches<int64_t>(lmax, 2, lmax / 2 + 1));
    EXPECT_FALSE(rocsparse::bsr_block_count_matches<int64_t>(lmax, 2, lmax / 2));
    EXPECT_TRUE(rocsparse::bsr_block_count_matches<int64_t>(lmax - 1, 2, lmax / 2));
    EXPECT_FALSE(rocsparse::bsr_block_count_matches<int64_t>(lmax, 2, lmax));
    EXPECT_TRUE(rocsparse::bsr_block_count_matches<int64_t>(lmax, 3, lmax / 3 + 1));
    EXPECT_TRUE(rocsparse::bsr_block_count_matches<int64_t>(lmax, lmax, 1));
    EXPECT_FALSE(rocsparse::bsr_block_count_matches<int64_t>(1, lmax, lmax));
    EXPECT_TRUE(rocsparse::bsr_block_count_matches<int64_t>(0, 0, lmax));
}

TEST(bsr_dims, diag_offset)
{
    // The last entry of bsr_val in the two bsrpad_value_extra cases. Without the cast on i
    // the first gives -2147386333; without the cast on block_index the second gives
    // -2143289345.
    EXPECT_EQ(rocsparse::bsr_diag_offset<int32_t>(0, 46341, 46342), 2147580963);
    EXPECT_EQ(rocsparse::bsr_diag_offset<int32_t>(512, 2047, 2048), 2151677951);
    EXPECT_EQ(rocsparse::bsr_diag_offset<int64_t>(0, 46341, 46342), 2147580963);
    EXPECT_EQ(rocsparse::bsr_diag_offset<int64_t>(512, 2047, 2048), 2151677951);

    EXPECT_EQ(rocsparse::bsr_diag_offset<int32_t>(0, 0, 1), 0);
    EXPECT_EQ(rocsparse::bsr_diag_offset<int32_t>(3, 1, 4), 3 * 16 + 1 * 4 + 1);
}

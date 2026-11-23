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

#pragma once

#include <cstdint>

namespace rocsparse
{
    // True iff mb == ceil(m / block_dim), i.e. (mb - 1) * block_dim < m <= mb * block_dim.
    // Evaluated without products so it cannot overflow at any index width. This equals
    // mb == rocsparse::ceil_div(m, block_dim) from rocsparse_grid.hpp, which this header
    // does not include so that the host-only unit test can use it.
    // Requires m >= 0, mb >= 0 and block_dim > 0.
    template <typename I>
    constexpr bool bsr_block_count_matches(I m, I mb, I block_dim)
    {
        return mb == m / block_dim + static_cast<I>(m % block_dim != 0);
    }

    // Offset of diagonal entry (i, i) of block block_index in bsr_val. Both terms are
    // widened before multiplying: BSR_IND_R computes i * block_dim in I, which overflows
    // 32 bits from block_dim = 46342.
    template <typename I>
    constexpr int64_t bsr_diag_offset(I block_index, I i, I block_dim)
    {
        return static_cast<int64_t>(block_index) * block_dim * block_dim
               + static_cast<int64_t>(i) * block_dim + i;
    }
}

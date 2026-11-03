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

#include "rocsparse_csrmv_adaptive_analysis.hpp"
#include "rocsparse_csrmv_partition.hpp"

template <typename I, typename J>
bool rocsparse::csrmv_partition<I, J>::is_poor_fit() const
{
    const I n_rows
        = this->row_ptr.empty() ? static_cast<I>(0) : static_cast<I>(this->row_ptr.size() - 1);
    if(this->row_blocks.size() < 2 || n_rows <= 0)
    {
        return false;
    }

    const std::size_t nblocks         = this->row_blocks.size() - 1;
    const I           span_limit      = static_cast<I>(csrmv_adaptive_wide_block_rows);
    const I           underfilled_nnz = static_cast<I>(csrmv_adaptive_block_nnz / 8);

    int64_t     wide_rows   = 0;
    std::size_t comparable  = 0;
    std::size_t underfilled = 0;

    for(std::size_t block = 0; block < nblocks; ++block)
    {
        I begin = this->row_blocks[block];
        I end   = this->row_blocks[block + 1];

        // The first block also takes the empty prefix adaptive does not skip,
        // and the last block the empty suffix. Count only the rows from the
        // first to the last nonempty one.
        if(block == 0)
        {
            while(begin < end && this->row_ptr[begin + 1] == this->row_ptr[begin])
            {
                ++begin;
            }
        }
        if(block == nblocks - 1)
        {
            while(end > begin && this->row_ptr[end] == this->row_ptr[end - 1])
            {
                --end;
            }
        }

        const I span = end - begin;
        if(span <= 0)
        {
            continue;
        }

        if(span > span_limit)
        {
            wide_rows += static_cast<int64_t>(span);
        }

        const I block_nnz = this->row_ptr[end] - this->row_ptr[begin];
        ++comparable;
        if(block_nnz < underfilled_nnz)
        {
            ++underfilled;
        }
    }

    // Wide blocks cover a substantial share of the matrix.
    if(wide_rows * 8 >= static_cast<int64_t>(n_rows))
    {
        return true;
    }

    // At least a quarter of the row blocks are short-row runs between long rows.
    return comparable >= 8 && underfilled * 4 >= comparable;
}

#define INSTANTIATE(I, J) template bool rocsparse::csrmv_partition<I, J>::is_poor_fit() const;

INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int64_t, int32_t);
INSTANTIATE(int64_t, int64_t);
#undef INSTANTIATE

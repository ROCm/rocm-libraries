/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_common.hpp"

namespace rocsparse
{
    ROCSPARSE_DEVICE_ILF void record_data_status(rocsparse_data_status* data_status,
                                                 rocsparse_data_status  status)
    {
        if(status != rocsparse_data_status_success)
        {
            *data_status = status;
        }
    }

    template <uint32_t BLOCKSIZE, typename T, typename I>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void check_matrix_ell_device(I m,
                                 I n,
                                 I ell_width,
                                 const T* __restrict__ ell_val,
                                 const I* __restrict__ ell_col_ind,
                                 rocsparse_index_base   idx_base,
                                 rocsparse_matrix_type  matrix_type,
                                 rocsparse_fill_mode    uplo,
                                 rocsparse_storage_mode storage,
                                 rocsparse_data_status* data_status)
    {
        const I row = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

        if(row >= m)
        {
            return;
        }

        // ELL element index m * j + row exceeds 32 bits once m * ell_width does
        // (a HYB matrix can have ell_nnz > INT32_MAX), so form it in 64-bit.
        for(I j = 0; j < ell_width; j++)
        {
            const int64_t index = static_cast<int64_t>(m) * j + row;
            const I       col   = ell_col_ind[index] - idx_base;

            if(ell_col_ind[index] == -1)
            {
                break;
            }

            // Check columns are in range [0...n)
            if(col < 0 || col >= n)
            {
                record_data_status(data_status, rocsparse_data_status_invalid_index);
                return;
            }

            const T val = ell_val[index];
            if(rocsparse::is_inf(val))
            {
                record_data_status(data_status, rocsparse_data_status_inf);
                return;
            }

            if(rocsparse::is_nan(val))
            {
                record_data_status(data_status, rocsparse_data_status_nan);
                return;
            }

            if(storage == rocsparse_storage_mode_sorted)
            {
                if(j > 0)
                {
                    const I prev_col = ell_col_ind[index - m] - idx_base;
                    if(prev_col == col)
                    {
                        record_data_status(data_status, rocsparse_data_status_duplicate_entry);
                        return;
                    }

                    if(prev_col > col)
                    {
                        record_data_status(data_status, rocsparse_data_status_invalid_sorting);
                        return;
                    }
                }
            }
        }
    }

    // Copies each row into a contiguous segment of zero-based column indices. Every slot
    // from the first -1 of a row onward is padding, stored as n so that it sorts last.
    template <uint32_t BLOCKSIZE, typename I>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void check_matrix_ell_gather_device(I m,
                                        I n,
                                        I ell_width,
                                        const I* __restrict__ in,
                                        rocsparse_index_base idx_base,
                                        I* __restrict__ out)
    {
        const int64_t row = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x;

        if(row >= m)
        {
            return;
        }

        bool padding = false;
        for(I j = 0; j < ell_width; ++j)
        {
            const I col = in[static_cast<int64_t>(m) * j + row];
            padding     = padding || (col == static_cast<I>(-1));

            out[row * ell_width + j] = padding ? n : static_cast<I>(col - idx_base);
        }
    }

    template <uint32_t BLOCKSIZE>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void check_matrix_ell_offsets_device(int64_t m, int64_t ell_width, int64_t* offsets)
    {
        const int64_t row = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x;

        if(row > m)
        {
            return;
        }

        offsets[row] = row * ell_width;
    }

    // Each row is a sorted segment of zero-based column indices followed by padding
    // stored as n.
    template <uint32_t BLOCKSIZE, typename I>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void check_matrix_ell_duplicate_device(
        I m, I n, I ell_width, const I* __restrict__ cols, rocsparse_data_status* data_status)
    {
        const int64_t row = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x;

        if(row >= m)
        {
            return;
        }

        const int64_t begin = row * ell_width;
        for(I j = 0; j + 1 < ell_width; ++j)
        {
            const I col = cols[begin + j];
            if(col == n)
            {
                return;
            }

            if(col == cols[begin + j + 1])
            {
                record_data_status(data_status, rocsparse_data_status_duplicate_entry);
                return;
            }
        }
    }
}

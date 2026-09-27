/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

    // Shift CSR offsets
    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void shift_offsets_kernel(J size, const I* __restrict__ in, I* __restrict__ out)
    {
        const J gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

        if(gid >= size)
        {
            return;
        }

        out[gid] = in[gid] - in[0];
    }

    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void check_row_ptr_array(J m,
                             const I* __restrict__ csr_row_ptr,
                             rocsparse_data_status* data_status)
    {
        const I gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

        if(gid < m)
        {
            const I start = csr_row_ptr[gid] - csr_row_ptr[0];
            const I end   = csr_row_ptr[gid + 1] - csr_row_ptr[0];

            if(start < 0 || end < 0)
            {
                record_data_status(data_status, rocsparse_data_status_invalid_offset_ptr);
                return;
            }

            if(end < start)
            {
                record_data_status(data_status, rocsparse_data_status_invalid_offset_ptr);
                return;
            }
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              bool     GRID_STRIDE,
              typename T,
              typename I,
              typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void check_matrix_csr_device(J m,
                                 J n,
                                 I nnz,
                                 const T* __restrict__ csr_val,
                                 const I* __restrict__ csr_row_ptr,
                                 const J*               csr_col_ind,
                                 const J*               csr_col_ind_sorted,
                                 rocsparse_index_base   idx_base,
                                 rocsparse_matrix_type  matrix_type,
                                 rocsparse_fill_mode    uplo,
                                 rocsparse_storage_mode storage,
                                 rocsparse_data_status* data_status)
    {
        // One WF_SIZE-wide lane group per row. BLOCKSIZE (256) is an exact multiple
        // of every dispatched WF_SIZE (4, 8, ..., 256), so the lane groups tile the
        // block and a block covers exactly ROWS_PER_BLOCK consecutive rows.
        static constexpr int ROWS_PER_BLOCK = static_cast<int>(BLOCKSIZE / WF_SIZE);

        const int tid = hipThreadIdx_x;
        const int lid = tid & static_cast<int>(WF_SIZE - 1);

        // Grid-stride over the rows. The launch has to clamp the block count (a
        // dispatch carries at most 2^32 - 1 work items, so a 256-thread block
        // permits only 16,777,215 blocks), which means one sweep of the grid is
        // not guaranteed to reach m and the previous `row = gid / WF_SIZE; if(row
        // >= m) return;` shape silently stopped at the end of the grid. At
        // m = 2^30 and WF_SIZE 4 that leaves the last 64 rows unexamined
        // (AISPARSE-698); the stride below closes the gap.
        //
        // `row_base` is int64_t because it is an induction variable that runs one
        // stride PAST m before the loop exits: with m near INT32_MAX and a stride
        // of hipGridDim_x * ROWS_PER_BLOCK rows, that final value does not fit in
        // int32_t. (The per-thread `row` itself always fit: `gid / WF_SIZE` in the
        // old code was evaluated in UNSIGNED arithmetic, because WF_SIZE is
        // uint32_t, so storing the thread id in a signed J was harmless.)
        //
        // Both `row_base` and the stride are derived only from hipBlockIdx_x,
        // hipGridDim_x, m and compile-time constants, i.e. they are block uniform,
        // so every thread of a block executes the same number of iterations. (The
        // `row >= m` tail and the error exits below are per-thread, but this
        // kernel contains no __syncthreads() and no wavefront collective, so that
        // divergence cannot deadlock.)
        for(int64_t row_base = static_cast<int64_t>(hipBlockIdx_x) * ROWS_PER_BLOCK; row_base < m;
            row_base += static_cast<int64_t>(hipGridDim_x) * ROWS_PER_BLOCK)
        {
            const int64_t row = row_base + tid / static_cast<int>(WF_SIZE);

            if(row >= m)
            {
                if constexpr(!GRID_STRIDE)
                {
                    break;
                }
                continue;
            }

            const I start = csr_row_ptr[row] - csr_row_ptr[0];
            const I end   = csr_row_ptr[row + 1] - csr_row_ptr[0];

            if(start < 0 || end < 0)
            {
                record_data_status(data_status, rocsparse_data_status_invalid_offset_ptr);
                return;
            }

            if(end < start)
            {
                record_data_status(data_status, rocsparse_data_status_invalid_offset_ptr);
                return;
            }

            for(I j = start + lid; j < end; j += WF_SIZE)
            {
                const J col = csr_col_ind[j] - idx_base;

                // Check columns are in range [0...n)
                if(col < 0 || col >= n)
                {
                    record_data_status(data_status, rocsparse_data_status_invalid_index);
                    return;
                }

                // Check that there are no duplicate columns
                if(j >= start + 1)
                {
                    const J scol      = csr_col_ind_sorted[j] - idx_base;
                    const J prev_scol = csr_col_ind_sorted[j - 1] - idx_base;

                    if(scol == prev_scol && (prev_scol >= 0 && prev_scol < n))
                    {
                        record_data_status(data_status, rocsparse_data_status_duplicate_entry);
                        return;
                    }
                }

                // check if values are inf or nan
                const T val = csr_val[j];
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

                // Check matrix type and fill mode is correct
                if(matrix_type != rocsparse_matrix_type_general)
                {
                    switch(uplo)
                    {
                    case rocsparse_fill_mode_lower:
                        if(row < col)
                        {
                            record_data_status(data_status, rocsparse_data_status_invalid_fill);
                            return;
                        }
                        break;
                    case rocsparse_fill_mode_upper:
                        if(row > col)
                        {
                            record_data_status(data_status, rocsparse_data_status_invalid_fill);
                            return;
                        }
                        break;
                    }
                }

                // Check sorting is correct
                if(storage == rocsparse_storage_mode_sorted)
                {
                    if(j >= start + 1)
                    {
                        const J prev_col = csr_col_ind[j - 1] - idx_base;

                        if(col <= prev_col && (prev_col >= 0 && prev_col < n))
                        {
                            record_data_status(data_status, rocsparse_data_status_invalid_sorting);
                            return;
                        }
                    }
                }
            }

            // A launch that did not clamp the block count already covers m in one
            // sweep, so the straight-line variant stops here instead of paying
            // the grid-stride loop's per-iteration control cost (AISPARSE-698).
            if constexpr(!GRID_STRIDE)
            {
                break;
            }
        }
    }
}

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

    // shift_offsets_kernel and check_row_ptr_array are launched with
    // GRID_STRIDE = true only when get_grid_size_x clamps grid.x below the
    // natural block count. The grid-stride index is int64_t because it runs one
    // stride past the end before the loop exits. An unclamped grid holds at most
    // 2^32 - 1 work items, so the straight-line variant's 32-bit thread id is
    // exact.

    // Shift CSR offsets.
    template <uint32_t BLOCKSIZE, bool GRID_STRIDE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void shift_offsets_kernel(J size, const I* __restrict__ in, I* __restrict__ out)
    {
        if constexpr(GRID_STRIDE)
        {
            for(int64_t gid = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x;
                gid < size;
                gid += static_cast<int64_t>(hipGridDim_x) * BLOCKSIZE)
            {
                out[gid] = in[gid] - in[0];
            }
        }
        else
        {
            const uint32_t gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

            if(static_cast<int64_t>(gid) < size)
            {
                out[gid] = in[gid] - in[0];
            }
        }
    }

    // Returns false, after recording the status, if row `row` has a negative or
    // decreasing offset.
    template <typename I, typename K>
    ROCSPARSE_DEVICE_ILF bool check_row_ptr_entry(K row,
                                                  const I* __restrict__ csr_row_ptr,
                                                  rocsparse_data_status* data_status)
    {
        const I start = csr_row_ptr[row] - csr_row_ptr[0];
        const I end   = csr_row_ptr[row + 1] - csr_row_ptr[0];

        if(start < 0 || end < 0 || end < start)
        {
            record_data_status(data_status, rocsparse_data_status_invalid_offset_ptr);
            return false;
        }
        return true;
    }

    template <uint32_t BLOCKSIZE, bool GRID_STRIDE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void check_row_ptr_array(J m,
                             const I* __restrict__ csr_row_ptr,
                             rocsparse_data_status* data_status)
    {
        if constexpr(GRID_STRIDE)
        {
            for(int64_t gid = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x;
                gid < m;
                gid += static_cast<int64_t>(hipGridDim_x) * BLOCKSIZE)
            {
                if(!check_row_ptr_entry(gid, csr_row_ptr, data_status))
                {
                    return;
                }
            }
        }
        else
        {
            const uint32_t gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

            if(static_cast<int64_t>(gid) < m)
            {
                check_row_ptr_entry(gid, csr_row_ptr, data_status);
            }
        }
    }

    // Checks row `row` (row < m) with one WF_SIZE-wide lane group; lane `lid`
    // takes entries start + lid, start + lid + WF_SIZE, ... Returns false, after
    // recording the status, as soon as the row is found invalid.
    template <uint32_t WF_SIZE, typename T, typename I, typename J>
    ROCSPARSE_DEVICE_ILF bool check_matrix_csr_row(J   row,
                                                   int lid,
                                                   J   n,
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
        const I start = csr_row_ptr[row] - csr_row_ptr[0];
        const I end   = csr_row_ptr[row + 1] - csr_row_ptr[0];

        if(start < 0 || end < 0)
        {
            record_data_status(data_status, rocsparse_data_status_invalid_offset_ptr);
            return false;
        }

        if(end < start)
        {
            record_data_status(data_status, rocsparse_data_status_invalid_offset_ptr);
            return false;
        }

        for(I j = start + lid; j < end; j += WF_SIZE)
        {
            const J col = csr_col_ind[j] - idx_base;

            // Check columns are in range [0...n)
            if(col < 0 || col >= n)
            {
                record_data_status(data_status, rocsparse_data_status_invalid_index);
                return false;
            }

            // Check that there are no duplicate columns
            if(j >= start + 1)
            {
                const J scol      = csr_col_ind_sorted[j] - idx_base;
                const J prev_scol = csr_col_ind_sorted[j - 1] - idx_base;

                if(scol == prev_scol && (prev_scol >= 0 && prev_scol < n))
                {
                    record_data_status(data_status, rocsparse_data_status_duplicate_entry);
                    return false;
                }
            }

            // check if values are inf or nan
            const T val = csr_val[j];
            if(rocsparse::is_inf(val))
            {
                record_data_status(data_status, rocsparse_data_status_inf);
                return false;
            }

            if(rocsparse::is_nan(val))
            {
                record_data_status(data_status, rocsparse_data_status_nan);
                return false;
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
                        return false;
                    }
                    break;
                case rocsparse_fill_mode_upper:
                    if(row > col)
                    {
                        record_data_status(data_status, rocsparse_data_status_invalid_fill);
                        return false;
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
                        return false;
                    }
                }
            }
        }
        return true;
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

        if constexpr(GRID_STRIDE)
        {
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
            // int32_t. A row that passes the `row >= m` test is below m, so it fits
            // in J.
            //
            // Both `row_base` and the stride are derived only from hipBlockIdx_x,
            // hipGridDim_x, m and compile-time constants, i.e. they are block uniform,
            // so every thread of a block executes the same number of iterations. (The
            // `row >= m` tail and the error exits are per-thread, but this kernel
            // contains no __syncthreads() and no wavefront collective, so that
            // divergence cannot deadlock.)
            for(int64_t row_base = static_cast<int64_t>(hipBlockIdx_x) * ROWS_PER_BLOCK;
                row_base < m;
                row_base += static_cast<int64_t>(hipGridDim_x) * ROWS_PER_BLOCK)
            {
                const int64_t row = row_base + tid / static_cast<int>(WF_SIZE);

                if(row >= m)
                {
                    continue;
                }

                if(!check_matrix_csr_row<WF_SIZE>(static_cast<J>(row),
                                                  lid,
                                                  n,
                                                  csr_val,
                                                  csr_row_ptr,
                                                  csr_col_ind,
                                                  csr_col_ind_sorted,
                                                  idx_base,
                                                  matrix_type,
                                                  uplo,
                                                  storage,
                                                  data_status))
                {
                    return;
                }
            }
        }
        else
        {
            // The launch did not clamp, so the grid covers m in one sweep and holds
            // at most 2^32 - 1 work items: the 32-bit thread id is exact, and
            // gid / WF_SIZE is below 2^31, so the row fits in J.
            static_assert(WF_SIZE >= 2, "the straight-line row index must fit in int32_t");

            const uint32_t gid = hipBlockIdx_x * BLOCKSIZE + tid;
            const J        row = static_cast<J>(gid / WF_SIZE);

            if(row >= m)
            {
                return;
            }

            check_matrix_csr_row<WF_SIZE>(row,
                                          lid,
                                          n,
                                          csr_val,
                                          csr_row_ptr,
                                          csr_col_ind,
                                          csr_col_ind_sorted,
                                          idx_base,
                                          matrix_type,
                                          uplo,
                                          storage,
                                          data_status);
        }
    }
}

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

    template <uint32_t BLOCKSIZE, typename T, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void check_matrix_coo_device(J       m,
                                 J       n,
                                 int64_t nnz,
                                 const T* __restrict__ coo_val,
                                 const I* __restrict__ coo_row_ind,
                                 const J* __restrict__ coo_col_ind,
                                 rocsparse_index_base   idx_base,
                                 rocsparse_matrix_type  matrix_type,
                                 rocsparse_fill_mode    uplo,
                                 rocsparse_storage_mode storage,
                                 rocsparse_data_status* data_status)
    {
        const int64_t gid = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x;

        if(gid >= nnz)
        {
            return;
        }

        // Check actual COO arrays for valid data
        const I row = coo_row_ind[gid] - idx_base;
        const I col = coo_col_ind[gid] - idx_base;

        if(row < 0 || row >= m)
        {
            record_data_status(data_status, rocsparse_data_status_invalid_index);
            return;
        }

        if(col < 0 || col >= n)
        {
            record_data_status(data_status, rocsparse_data_status_invalid_index);
            return;
        }

        // check if values are inf or nan
        const T val = coo_val[gid];
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
            if(gid < nnz - 1)
            {
                const I    next_row    = coo_row_ind[gid + 1] - idx_base;
                const I    next_col    = coo_col_ind[gid + 1] - idx_base;
                const bool next_row_ok = (next_row >= 0 && next_row < m);
                const bool next_col_ok = (next_col >= 0 && next_col < n);

                if(next_row_ok && row > next_row)
                {
                    record_data_status(data_status, rocsparse_data_status_invalid_sorting);
                    return;
                }

                if(next_row_ok && row == next_row && next_col_ok)
                {
                    if(col == next_col)
                    {
                        record_data_status(data_status, rocsparse_data_status_duplicate_entry);
                        return;
                    }

                    if(col > next_col)
                    {
                        record_data_status(data_status, rocsparse_data_status_invalid_sorting);
                        return;
                    }
                }
            }
        }
    }

    // Entries gid and gid + 1 of a copy sorted by row, then by column.
    template <typename I>
    ROCSPARSE_DEVICE_ILF void check_matrix_coo_duplicate_pair(int64_t gid,
                                                              I       m,
                                                              I       n,
                                                              const I* __restrict__ rows,
                                                              const I* __restrict__ cols,
                                                              rocsparse_index_base   idx_base,
                                                              rocsparse_data_status* data_status)
    {
        const I row = rows[gid] - idx_base;
        const I col = cols[gid] - idx_base;

        if(row < 0 || row >= m || col < 0 || col >= n)
        {
            return;
        }

        const I next_row = rows[gid + 1] - idx_base;
        const I next_col = cols[gid + 1] - idx_base;

        if(row == next_row && col == next_col)
        {
            record_data_status(data_status, rocsparse_data_status_duplicate_entry);
        }
    }

    // Launched with GRID_STRIDE = true only when get_grid_size_x clamps grid.x below the
    // natural block count.
    template <uint32_t BLOCKSIZE, bool GRID_STRIDE, typename I>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void check_matrix_coo_duplicate_device(I       m,
                                           I       n,
                                           int64_t nnz,
                                           const I* __restrict__ rows,
                                           const I* __restrict__ cols,
                                           rocsparse_index_base   idx_base,
                                           rocsparse_data_status* data_status)
    {
        if constexpr(GRID_STRIDE)
        {
            for(int64_t gid = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x;
                gid < nnz - 1;
                gid += static_cast<int64_t>(hipGridDim_x) * BLOCKSIZE)
            {
                check_matrix_coo_duplicate_pair(gid, m, n, rows, cols, idx_base, data_status);
            }
        }
        else
        {
            const int64_t gid = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE + hipThreadIdx_x;

            if(gid < nnz - 1)
            {
                check_matrix_coo_duplicate_pair(gid, m, n, rows, cols, idx_base, data_status);
            }
        }
    }
}

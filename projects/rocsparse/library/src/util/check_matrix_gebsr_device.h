/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "common.h"

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

    // Shift GEBSR offsets
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
    void check_row_ptr_array(J mb,
                             const I* __restrict__ bsr_row_ptr,
                             rocsparse_data_status* data_status)
    {
        const I gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

        if(gid < mb)
        {
            const I start = bsr_row_ptr[gid] - bsr_row_ptr[0];
            const I end   = bsr_row_ptr[gid + 1] - bsr_row_ptr[0];

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

    template <uint32_t BLOCKSIZE, typename T, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void check_matrix_gebsr_device(rocsparse_direction dir,
                                   J                   mb,
                                   J                   nb,
                                   I                   nnzb,
                                   J                   row_block_dim,
                                   J                   col_block_dim,
                                   const T* __restrict__ bsr_val,
                                   const I* __restrict__ bsr_row_ptr,
                                   const J*               bsr_col_ind,
                                   const J*               bsr_col_ind_sorted,
                                   rocsparse_index_base   idx_base,
                                   rocsparse_matrix_type  matrix_type,
                                   rocsparse_fill_mode    uplo,
                                   rocsparse_storage_mode storage,
                                   rocsparse_data_status* data_status)
    {
        const I row = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

        if(row >= mb)
        {
            return;
        }

        const I start = bsr_row_ptr[row] - bsr_row_ptr[0];
        const I end   = bsr_row_ptr[row + 1] - bsr_row_ptr[0];

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

        for(I j = start; j < end; j++)
        {
            const J col = bsr_col_ind[j] - idx_base;

            // Check columns are in range [0...nb)
            if(col < 0 || col >= nb)
            {
                record_data_status(data_status, rocsparse_data_status_invalid_index);
                return;
            }

            // Check that there are no duplicate columns
            if(j >= start + 1)
            {
                const J scol      = bsr_col_ind_sorted[j] - idx_base;
                const J prev_scol = bsr_col_ind_sorted[j - 1] - idx_base;

                if(scol == prev_scol && (prev_scol >= 0 && prev_scol < nb))
                {
                    record_data_status(data_status, rocsparse_data_status_duplicate_entry);
                    return;
                }
            }

            // check if values are inf or nan
            if(dir == rocsparse_direction_row)
            {
                for(J r = 0; r < row_block_dim; r++)
                {
                    for(J c = 0; c < col_block_dim; c++)
                    {
                        const T val = bsr_val[int64_t(row_block_dim) * col_block_dim * j
                                              + col_block_dim * r + c];
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
                    }
                }
            }
            else
            {
                for(J c = 0; c < col_block_dim; c++)
                {
                    for(J r = 0; r < row_block_dim; r++)
                    {
                        const T val = bsr_val[int64_t(row_block_dim) * col_block_dim * j
                                              + row_block_dim * c + r];
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
                    }
                }
            }

            // Check matrix type and fill mode is correct
            if(matrix_type != rocsparse_matrix_type_general)
            {
                if(uplo == rocsparse_fill_mode_lower)
                {
                    if(row < col)
                    {
                        record_data_status(data_status, rocsparse_data_status_invalid_fill);
                        return;
                    }
                }
                else
                {
                    if(row > col)
                    {
                        record_data_status(data_status, rocsparse_data_status_invalid_fill);
                        return;
                    }
                }
            }

            // Check sorting is correct
            if(storage == rocsparse_storage_mode_sorted)
            {
                if(j >= start + 1)
                {
                    const J prev_col = bsr_col_ind[j - 1] - idx_base;

                    if(col <= prev_col && (prev_col >= 0 && prev_col < nb))
                    {
                        record_data_status(data_status, rocsparse_data_status_invalid_sorting);
                        return;
                    }
                }
            }
        }
    }
}

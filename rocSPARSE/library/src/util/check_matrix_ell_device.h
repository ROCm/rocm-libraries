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

        for(I j = 0; j < ell_width; j++)
        {
            const I col = ell_col_ind[m * j + row] - idx_base;

            if(ell_col_ind[m * j + row] == -1)
            {
                break;
            }

            // Check columns are in range [0...n)
            if(col < 0 || col >= n)
            {
                record_data_status(data_status, rocsparse_data_status_invalid_index);
                return;
            }

            const T val = ell_val[m * j + row];
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
                    const I prev_col = ell_col_ind[m * (j - 1) + row] - idx_base;
                    if(prev_col >= col)
                    {
                        record_data_status(data_status, rocsparse_data_status_invalid_sorting);
                        return;
                    }
                }
            }
        }
    }
}

/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "dense2csx_device.h"

namespace rocsparse
{
    template <rocsparse_direction DIRA, typename I, typename J, typename T>
    rocsparse_status dense2csx_template(rocsparse_handle          handle,
                                        rocsparse_order           order,
                                        J                         m,
                                        J                         n,
                                        const rocsparse_mat_descr descr,
                                        const T*                  A,
                                        int64_t                   ld,
                                        T*                        csx_val,
                                        I*                        csx_row_col_ptr,
                                        J*                        csx_col_row_ind)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if(0 == m || 0 == n)
        {
            return rocsparse_status_success;
        }

        static constexpr rocsparse_int data_ratio = sizeof(T) / sizeof(float);
        hipStream_t                    stream     = handle->stream;
        switch(DIRA)
        {
        case rocsparse_direction_row:
        {
            if(handle->wavefront_size == 32)
            {
                static constexpr rocsparse_int WF_SIZE = 32;
                static constexpr rocsparse_int NROWS_PER_BLOCK
                    = 16 / (data_ratio > 0 ? data_ratio : 1);
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::dense2csr_kernel<NROWS_PER_BLOCK, WF_SIZE>),
                    dim3((m - 1) / NROWS_PER_BLOCK + 1),
                    dim3(WF_SIZE * NROWS_PER_BLOCK),
                    0,
                    stream,
                    descr->base,
                    order,
                    m,
                    n,
                    A,
                    ld,
                    csx_val,
                    csx_row_col_ptr,
                    csx_col_row_ind);
            }
            else
            {
                static constexpr rocsparse_int WF_SIZE = 64;
                static constexpr rocsparse_int NROWS_PER_BLOCK
                    = 16 / (data_ratio > 0 ? data_ratio : 1);
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::dense2csr_kernel<NROWS_PER_BLOCK, WF_SIZE>),
                    dim3((m - 1) / NROWS_PER_BLOCK + 1),
                    dim3(WF_SIZE * NROWS_PER_BLOCK),
                    0,
                    stream,
                    descr->base,
                    order,
                    m,
                    n,
                    A,
                    ld,
                    csx_val,
                    csx_row_col_ptr,
                    csx_col_row_ind);
            }

            return rocsparse_status_success;
        }

        case rocsparse_direction_column:
        {
            if(handle->wavefront_size == 32)
            {
                static constexpr rocsparse_int WF_SIZE = 32;
                static constexpr rocsparse_int NCOLUMNS_PER_BLOCK
                    = 16 / (data_ratio > 0 ? data_ratio : 1);
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::dense2csc_kernel<NCOLUMNS_PER_BLOCK, WF_SIZE>),
                    dim3((n - 1) / NCOLUMNS_PER_BLOCK + 1),
                    dim3(WF_SIZE * NCOLUMNS_PER_BLOCK),
                    0,
                    stream,
                    descr->base,
                    order,
                    m,
                    n,
                    A,
                    ld,
                    csx_val,
                    csx_row_col_ptr,
                    csx_col_row_ind);
            }
            else
            {
                static constexpr rocsparse_int WF_SIZE = 64;
                static constexpr rocsparse_int NCOLUMNS_PER_BLOCK
                    = 16 / (data_ratio > 0 ? data_ratio : 1);
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::dense2csc_kernel<NCOLUMNS_PER_BLOCK, WF_SIZE>),
                    dim3((n - 1) / NCOLUMNS_PER_BLOCK + 1),
                    dim3(WF_SIZE * NCOLUMNS_PER_BLOCK),
                    0,
                    stream,
                    descr->base,
                    order,
                    m,
                    n,
                    A,
                    ld,
                    csx_val,
                    csx_row_col_ptr,
                    csx_col_row_ind);
            }

            return rocsparse_status_success;
        }
        }

        // LCOV_EXCL_START
        return rocsparse_status_invalid_value;
        // LCOV_EXCL_STOP
    }
}

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

#include "csx2dense_device.h"

namespace rocsparse
{
    template <rocsparse_direction DIRA, typename I, typename J, typename T>
    rocsparse_status csx2dense_template(rocsparse_handle          handle,
                                        J                         m,
                                        J                         n,
                                        const rocsparse_mat_descr descr,
                                        const T*                  csx_val,
                                        const I*                  csx_row_col_ptr,
                                        const J*                  csx_col_row_ind,
                                        T*                        A,
                                        int64_t                   ld,
                                        rocsparse_order           order)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if(0 == m || 0 == n)
        {
            return rocsparse_status_success;
        }

        hipStream_t stream = handle->stream;

        switch(DIRA)
        {
        case rocsparse_direction_row:
        {
            if(handle->wavefront_size == 32)
            {
                static constexpr rocsparse_int WAVEFRONT_SIZE  = 32;
                static constexpr rocsparse_int NROWS_PER_BLOCK = 16;

                rocsparse_int blocks = (m - 1) / NROWS_PER_BLOCK + 1;
                dim3          k_blocks(blocks), k_threads(WAVEFRONT_SIZE * NROWS_PER_BLOCK);

                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csr2dense_kernel<NROWS_PER_BLOCK, WAVEFRONT_SIZE, I, J, T>),
                    k_blocks,
                    k_threads,
                    0,
                    stream,
                    descr->base,
                    m,
                    n,
                    csx_val,
                    csx_row_col_ptr,
                    csx_col_row_ind,
                    A,
                    ld,
                    order);
            }
            else
            {
                static constexpr rocsparse_int WAVEFRONT_SIZE  = 64;
                static constexpr rocsparse_int NROWS_PER_BLOCK = 16;

                rocsparse_int blocks = (m - 1) / NROWS_PER_BLOCK + 1;
                dim3          k_blocks(blocks), k_threads(WAVEFRONT_SIZE * NROWS_PER_BLOCK);

                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csr2dense_kernel<NROWS_PER_BLOCK, WAVEFRONT_SIZE, I, J, T>),
                    k_blocks,
                    k_threads,
                    0,
                    stream,
                    descr->base,
                    m,
                    n,
                    csx_val,
                    csx_row_col_ptr,
                    csx_col_row_ind,
                    A,
                    ld,
                    order);
            }

            return rocsparse_status_success;
        }

        case rocsparse_direction_column:
        {
            if(handle->wavefront_size == 32)
            {
                static constexpr rocsparse_int WAVEFRONT_SIZE     = 32;
                static constexpr rocsparse_int NCOLUMNS_PER_BLOCK = 16;

                rocsparse_int blocks = (n - 1) / NCOLUMNS_PER_BLOCK + 1;
                dim3          k_blocks(blocks), k_threads(WAVEFRONT_SIZE * NCOLUMNS_PER_BLOCK);

                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csc2dense_kernel<NCOLUMNS_PER_BLOCK, WAVEFRONT_SIZE, I, J, T>),
                    k_blocks,
                    k_threads,
                    0,
                    stream,
                    descr->base,
                    m,
                    n,
                    csx_val,
                    csx_row_col_ptr,
                    csx_col_row_ind,
                    A,
                    ld,
                    order);
            }
            else
            {
                static constexpr rocsparse_int WAVEFRONT_SIZE     = 64;
                static constexpr rocsparse_int NCOLUMNS_PER_BLOCK = 16;

                rocsparse_int blocks = (n - 1) / NCOLUMNS_PER_BLOCK + 1;
                dim3          k_blocks(blocks), k_threads(WAVEFRONT_SIZE * NCOLUMNS_PER_BLOCK);

                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::csc2dense_kernel<NCOLUMNS_PER_BLOCK, WAVEFRONT_SIZE, I, J, T>),
                    k_blocks,
                    k_threads,
                    0,
                    stream,
                    descr->base,
                    m,
                    n,
                    csx_val,
                    csx_row_col_ptr,
                    csx_col_row_ind,
                    A,
                    ld,
                    order);
            }

            return rocsparse_status_success;
        }
        }

        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }
}

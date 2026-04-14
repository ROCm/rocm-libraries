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

#include "internal/conversion/rocsparse_bsrpad_value.h"
#include "rocsparse_bsrpad_value.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

#include "bsrpad_value_device.h"

template <typename T>
rocsparse_status rocsparse::bsrpad_value_template(rocsparse_handle          handle, //0
                                                  rocsparse_int             m, //1
                                                  rocsparse_int             mb, //2
                                                  rocsparse_int             nnzb, //3
                                                  rocsparse_int             block_dim, //4
                                                  T                         value, //5
                                                  const rocsparse_mat_descr bsr_descr, //6
                                                  T*                        bsr_val, //7
                                                  const rocsparse_int*      bsr_row_ptr, //8
                                                  const rocsparse_int*      bsr_col_ind) //9
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xbsrpad_value"),
                         m,
                         mb,
                         nnzb,
                         block_dim,
                         value,
                         bsr_descr,
                         (const void*&)bsr_val,
                         (const void*&)bsr_row_ptr,
                         (const void*&)bsr_col_ind);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, mb);
    ROCSPARSE_CHECKARG_SIZE(3, nnzb);
    ROCSPARSE_CHECKARG_SIZE(4, block_dim);
    ROCSPARSE_CHECKARG(4, block_dim, (block_dim == 0), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_POINTER(6, bsr_descr);
    ROCSPARSE_CHECKARG_ARRAY(7, nnzb, bsr_val);
    ROCSPARSE_CHECKARG_ARRAY(8, mb, bsr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(9, nnzb, bsr_col_ind);
    ROCSPARSE_CHECKARG(1, m, (mb * block_dim < m), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG(1, m, (m <= (mb - 1) * block_dim), rocsparse_status_invalid_size);

    // Quick return if possible
    if(mb == 0 || nnzb == 0)
    {
        return rocsparse_status_success;
    }

    // Quick return if possible
    if(mb * block_dim == m)
    {
        return rocsparse_status_success;
    }

    constexpr rocsparse_int block_size = 1024;
    const rocsparse_int     grid_size  = (block_dim + block_size - 1) / block_size;

    // Check matrix sorting mode
    if(bsr_descr->storage_mode == rocsparse_storage_mode_sorted)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::bsrpad_value_kernel_sorted<block_size>),
                                           dim3(grid_size),
                                           dim3(block_size),
                                           0,
                                           handle->stream,
                                           m,
                                           mb,
                                           block_dim,
                                           value,
                                           bsr_descr->base,
                                           bsr_val,
                                           bsr_row_ptr,
                                           bsr_col_ind);
    }
    else
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::bsrpad_value_kernel_unsorted<block_size>),
                                           dim3(grid_size),
                                           dim3(block_size),
                                           0,
                                           handle->stream,
                                           m,
                                           mb,
                                           block_dim,
                                           value,
                                           bsr_descr->base,
                                           bsr_val,
                                           bsr_row_ptr,
                                           bsr_col_ind);
    }

    return rocsparse_status_success;
}

//
// C INTERFACE
//
#define C_IMPL(NAME, TYPE)                                                                         \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,                             \
                                     rocsparse_int             m,                                  \
                                     rocsparse_int             mb,                                 \
                                     rocsparse_int             nnzb,                               \
                                     rocsparse_int             block_dim,                          \
                                     TYPE                      value,                              \
                                     const rocsparse_mat_descr bsr_descr,                          \
                                     TYPE*                     bsr_val,                            \
                                     const rocsparse_int*      bsr_row_ptr,                        \
                                     const rocsparse_int*      bsr_col_ind)                        \
                                                                                                   \
    try                                                                                            \
    {                                                                                              \
        ROCSPARSE_ROUTINE_TRACE;                                                                   \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrpad_value_template<TYPE>(                          \
            handle, m, mb, nnzb, block_dim, value, bsr_descr, bsr_val, bsr_row_ptr, bsr_col_ind)); \
        return rocsparse_status_success;                                                           \
    }                                                                                              \
    catch(...)                                                                                     \
    {                                                                                              \
        RETURN_ROCSPARSE_EXCEPTION();                                                              \
    }

C_IMPL(rocsparse_sbsrpad_value, float);
C_IMPL(rocsparse_dbsrpad_value, double);
C_IMPL(rocsparse_cbsrpad_value, rocsparse_float_complex);
C_IMPL(rocsparse_zbsrpad_value, rocsparse_double_complex);
#undef C_IMPL

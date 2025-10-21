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

#include "internal/level2/rocsparse_bsrsv.h"
#include "rocsparse_bsrsv.hpp"
#include "rocsparse_csrsv.hpp"
#include "rocsparse_templates.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <typename T>
    static rocsparse_status bsrsv_buffer_size_impl(rocsparse_handle          handle,
                                                   rocsparse_direction       dir,
                                                   rocsparse_operation       trans,
                                                   rocsparse_int             mb,
                                                   rocsparse_int             nnzb,
                                                   const rocsparse_mat_descr descr,
                                                   const T*                  bsr_val,
                                                   const rocsparse_int*      bsr_row_ptr,
                                                   const rocsparse_int*      bsr_col_ind,
                                                   rocsparse_int             block_dim,
                                                   rocsparse_mat_info        info,
                                                   size_t*                   buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Check for valid handle and matrix descriptor
        ROCSPARSE_CHECKARG_HANDLE(0, handle);

        // Logging
        rocsparse::log_trace(handle,
                             rocsparse::replaceX<T>("rocsparse_Xbsrsv_buffer_size"),
                             dir,
                             trans,
                             mb,
                             nnzb,
                             (const void*&)descr,
                             (const void*&)bsr_val,
                             (const void*&)bsr_row_ptr,
                             (const void*&)bsr_col_ind,
                             block_dim,
                             (const void*&)info,
                             (const void*&)buffer_size);

        ROCSPARSE_CHECKARG_ENUM(1, dir);
        ROCSPARSE_CHECKARG_ENUM(2, trans);
        ROCSPARSE_CHECKARG_SIZE(3, mb);
        ROCSPARSE_CHECKARG_SIZE(4, nnzb);

        ROCSPARSE_CHECKARG_POINTER(5, descr);
        ROCSPARSE_CHECKARG(5,
                           descr,
                           (descr->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(5,
                           descr,
                           (descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);

        ROCSPARSE_CHECKARG_ARRAY(6, nnzb, bsr_val);
        ROCSPARSE_CHECKARG_ARRAY(7, mb, bsr_row_ptr);
        ROCSPARSE_CHECKARG_ARRAY(8, nnzb, bsr_col_ind);
        ROCSPARSE_CHECKARG(9, block_dim, (block_dim <= 0), rocsparse_status_invalid_size);
        ROCSPARSE_CHECKARG_POINTER(10, info);
        ROCSPARSE_CHECKARG_POINTER(11, buffer_size);

        RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrsv_buffer_size_template<rocsparse_int,
                                                                         rocsparse_int,
                                                                         T>(
            handle, trans, mb, nnzb, descr, bsr_val, bsr_row_ptr, bsr_col_ind, info, buffer_size)));

        if(trans == rocsparse_operation_transpose)
        {
            /* Remove additional CSR buffer */
            *buffer_size -= ((sizeof(T) * nnzb - 1) / 256 + 1) * 256;
            /* Add BSR buffer instead */
            *buffer_size += ((sizeof(T) * nnzb * block_dim * block_dim - 1) / 256 + 1) * 256;
        }

        return rocsparse_status_success;
    }
}

#define C_IMPL(NAME, TYPE)                                                         \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,             \
                                     rocsparse_direction       dir,                \
                                     rocsparse_operation       trans,              \
                                     rocsparse_int             mb,                 \
                                     rocsparse_int             nnzb,               \
                                     const rocsparse_mat_descr descr,              \
                                     const TYPE*               bsr_val,            \
                                     const rocsparse_int*      bsr_row_ptr,        \
                                     const rocsparse_int*      bsr_col_ind,        \
                                     rocsparse_int             block_dim,          \
                                     rocsparse_mat_info        info,               \
                                     size_t*                   buffer_size)        \
    try                                                                            \
    {                                                                              \
        ROCSPARSE_ROUTINE_TRACE;                                                   \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrsv_buffer_size_impl(handle,        \
                                                                    dir,           \
                                                                    trans,         \
                                                                    mb,            \
                                                                    nnzb,          \
                                                                    descr,         \
                                                                    bsr_val,       \
                                                                    bsr_row_ptr,   \
                                                                    bsr_col_ind,   \
                                                                    block_dim,     \
                                                                    info,          \
                                                                    buffer_size)); \
        return rocsparse_status_success;                                           \
    }                                                                              \
    catch(...)                                                                     \
    {                                                                              \
        RETURN_ROCSPARSE_EXCEPTION();                                              \
    }

C_IMPL(rocsparse_sbsrsv_buffer_size, float);
C_IMPL(rocsparse_dbsrsv_buffer_size, double);
C_IMPL(rocsparse_cbsrsv_buffer_size, rocsparse_float_complex);
C_IMPL(rocsparse_zbsrsv_buffer_size, rocsparse_double_complex);
#undef C_IMPL

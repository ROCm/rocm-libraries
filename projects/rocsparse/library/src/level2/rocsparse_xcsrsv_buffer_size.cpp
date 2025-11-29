/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/level2/rocsparse_csrsv.h"
#include "rocsparse_control.hpp"
#include "rocsparse_csrsv.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
namespace rocsparse
{
    static rocsparse_status xcsrsv_buffer_size_checkarg(rocsparse_handle          handle, // 0
                                                        rocsparse_operation       trans, // 1
                                                        int64_t                   m, // 2
                                                        int64_t                   nnz, // 3
                                                        const rocsparse_mat_descr descr, // 4
                                                        const void*               csr_val, // 5
                                                        const void*               csr_row_ptr, // 6
                                                        const void*               csr_col_ind, // 7
                                                        rocsparse_mat_info        info, // 8
                                                        size_t*                   buffer_size // 9
    )
    {
        ROCSPARSE_ROUTINE_TRACE;
        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_ENUM(1, trans);
        ROCSPARSE_CHECKARG_SIZE(2, m);
        ROCSPARSE_CHECKARG_SIZE(3, nnz);
        ROCSPARSE_CHECKARG_POINTER(4, descr);
        ROCSPARSE_CHECKARG_ARRAY(5, nnz, csr_val);
        ROCSPARSE_CHECKARG_ARRAY(6, m, csr_row_ptr);
        ROCSPARSE_CHECKARG_ARRAY(7, nnz, csr_col_ind);
        ROCSPARSE_CHECKARG_POINTER(8, info);
        ROCSPARSE_CHECKARG_POINTER(9, buffer_size);
        return rocsparse_status_success;
    }

    template <typename T>
    static rocsparse_status xcsrsv_buffer_size(rocsparse_handle          handle,
                                               rocsparse_operation       trans,
                                               rocsparse_int             m,
                                               rocsparse_int             nnz,
                                               const rocsparse_mat_descr descr,
                                               const void*               csr_val,
                                               const void*               csr_row_ptr,
                                               const void*               csr_col_ind,
                                               rocsparse_mat_info        info,
                                               size_t*                   buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;
        if(m == 0)
        {
            *buffer_size = 0;
            return rocsparse_status_success;
        }
        RETURN_IF_ROCSPARSE_ERROR(xcsrsv_buffer_size_checkarg(
            handle, trans, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size));

        _rocsparse_spmat_descr csr(rocsparse_format_csr,

                                   false,
                                   static_cast<int64_t>(1),
                                   m,
                                   m,
                                   nnz,

                                   rocsparse::get_datatype<T>(),
                                   csr_val,
                                   nullptr,
                                   static_cast<int64_t>(0),

                                   //
                                   rocsparse::get_indextype<rocsparse_int>(),
                                   csr_row_ptr,
                                   nullptr,
                                   static_cast<int64_t>(0),

                                   rocsparse::get_indextype<rocsparse_int>(),
                                   csr_col_ind,
                                   nullptr,
                                   static_cast<int64_t>(0),

                                   descr->base,
                                   descr,
                                   info);
        size_t                 buffer_sizes[2]{};
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::csrsv_analysis_buffer_size(handle, trans, &csr, &buffer_sizes[0]));

        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::csrsv_solve_buffer_size(handle, trans, &csr, &buffer_sizes[1]));

        buffer_size[0] = std::max(buffer_sizes[0], buffer_sizes[1]);
        return rocsparse_status_success;
    }

}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
#define CIMPL(NAME, T)                                                                             \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,                             \
                                     rocsparse_operation       trans,                              \
                                     rocsparse_int             m,                                  \
                                     rocsparse_int             nnz,                                \
                                     const rocsparse_mat_descr descr,                              \
                                     const T*                  csr_val,                            \
                                     const rocsparse_int*      csr_row_ptr,                        \
                                     const rocsparse_int*      csr_col_ind,                        \
                                     rocsparse_mat_info        info,                               \
                                     size_t*                   buffer_size)                        \
    try                                                                                            \
    {                                                                                              \
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::xcsrsv_buffer_size<T>(                               \
            handle, trans, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size))); \
        return rocsparse_status_success;                                                           \
    }                                                                                              \
    catch(...)                                                                                     \
    {                                                                                              \
        RETURN_ROCSPARSE_EXCEPTION();                                                              \
    }

CIMPL(rocsparse_scsrsv_buffer_size, float);
CIMPL(rocsparse_dcsrsv_buffer_size, double);
CIMPL(rocsparse_ccsrsv_buffer_size, rocsparse_float_complex);
CIMPL(rocsparse_zcsrsv_buffer_size, rocsparse_double_complex);

#undef CIMPL

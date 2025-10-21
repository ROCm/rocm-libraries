/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "../level2/rocsparse_csrsv.hpp"
#include "internal/precond/rocsparse_csrilu0.h"
#include "rocsparse_csrilu0.hpp"

namespace rocsparse
{
    template <typename T>
    static rocsparse_status csrilu0_buffer_size_core(rocsparse_handle          handle,
                                                     rocsparse_int             m,
                                                     rocsparse_int             nnz,
                                                     const rocsparse_mat_descr descr,
                                                     const T*                  csr_val,
                                                     const rocsparse_int*      csr_row_ptr,
                                                     const rocsparse_int*      csr_col_ind,
                                                     rocsparse_mat_info        info,
                                                     size_t*                   buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;

        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::csrsv_buffer_size_template<rocsparse_int, rocsparse_int, T>(
                handle,
                rocsparse_operation_none,
                m,
                nnz,
                descr,
                csr_val,
                csr_row_ptr,
                csr_col_ind,
                info,
                buffer_size)));
        return rocsparse_status_success;
    }

    static rocsparse_status csrilu0_buffer_size_quickreturn(rocsparse_handle          handle,
                                                            int64_t                   m,
                                                            int64_t                   nnz,
                                                            const rocsparse_mat_descr descr,
                                                            const void*               csr_val,
                                                            const void*               csr_row_ptr,
                                                            const void*               csr_col_ind,
                                                            rocsparse_mat_info        info,
                                                            size_t*                   buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;

        return rocsparse_status_continue;
    }

    static rocsparse_status csrilu0_buffer_size_checkarg(rocsparse_handle          handle,
                                                         int64_t                   m,
                                                         int64_t                   nnz,
                                                         const rocsparse_mat_descr descr,
                                                         const void*               csr_val,
                                                         const void*               csr_row_ptr,
                                                         const void*               csr_col_ind,
                                                         rocsparse_mat_info        info,
                                                         size_t*                   buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_SIZE(1, m);

        const rocsparse_status status = rocsparse::csrilu0_buffer_size_quickreturn(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG_SIZE(2, nnz);
        ROCSPARSE_CHECKARG_POINTER(3, descr);
        ROCSPARSE_CHECKARG(3,
                           descr,
                           (descr->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(3,
                           descr,
                           (descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);
        ROCSPARSE_CHECKARG_ARRAY(4, nnz, csr_val);
        ROCSPARSE_CHECKARG_ARRAY(5, m, csr_row_ptr);
        ROCSPARSE_CHECKARG_ARRAY(6, nnz, csr_col_ind);
        ROCSPARSE_CHECKARG_POINTER(7, info);
        ROCSPARSE_CHECKARG_POINTER(8, buffer_size);
        return rocsparse_status_continue;
    }

    template <typename T>
    static rocsparse_status csrilu0_buffer_size_impl(rocsparse_handle          handle,
                                                     rocsparse_int             m,
                                                     rocsparse_int             nnz,
                                                     const rocsparse_mat_descr descr,
                                                     const T*                  csr_val,
                                                     const rocsparse_int*      csr_row_ptr,
                                                     const rocsparse_int*      csr_col_ind,
                                                     rocsparse_mat_info        info,
                                                     size_t*                   buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;

        rocsparse::log_trace(handle,
                             rocsparse::replaceX<T>("rocsparse_Xcsrilu0_buffer_size"),
                             m,
                             nnz,
                             (const void*&)descr,
                             (const void*&)csr_val,
                             (const void*&)csr_row_ptr,
                             (const void*&)csr_col_ind,
                             (const void*&)info,
                             (const void*&)buffer_size);

        const rocsparse_status status = rocsparse::csrilu0_buffer_size_checkarg(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrilu0_buffer_size_core(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size));

        return rocsparse_status_success;
    }
}

#define CIMPL(NAME, T)                                                                     \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,                     \
                                     rocsparse_int             m,                          \
                                     rocsparse_int             nnz,                        \
                                     const rocsparse_mat_descr descr,                      \
                                     const T*                  csr_val,                    \
                                     const rocsparse_int*      csr_row_ptr,                \
                                     const rocsparse_int*      csr_col_ind,                \
                                     rocsparse_mat_info        info,                       \
                                     size_t*                   buffer_size)                \
    try                                                                                    \
    {                                                                                      \
        ROCSPARSE_ROUTINE_TRACE;                                                           \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrilu0_buffer_size_impl(                     \
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, buffer_size)); \
        return rocsparse_status_success;                                                   \
    }                                                                                      \
    catch(...)                                                                             \
    {                                                                                      \
        RETURN_ROCSPARSE_EXCEPTION();                                                      \
    }

CIMPL(rocsparse_scsrilu0_buffer_size, float);
CIMPL(rocsparse_dcsrilu0_buffer_size, double);
CIMPL(rocsparse_ccsrilu0_buffer_size, rocsparse_float_complex);
CIMPL(rocsparse_zcsrilu0_buffer_size, rocsparse_double_complex);
#undef CIMPL

/*! \file */
/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/precond/rocsparse_csrilu0.h"
#include "rocsparse_csrilu0.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    static rocsparse_status xcsrilu0_checkarg(rocsparse_handle          handle, //0
                                              int64_t                   m, //1
                                              int64_t                   nnz, //2
                                              const rocsparse_mat_descr descr, //3
                                              void*                     csr_val, //4
                                              const void*               csr_row_ptr, //5
                                              const void*               csr_col_ind, //6
                                              rocsparse_mat_info        info, //7
                                              rocsparse_solve_policy    policy, //8
                                              void*                     temp_buffer) //9
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_SIZE(1, m);

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
        ROCSPARSE_CHECKARG_ENUM(8, policy);
        ROCSPARSE_CHECKARG_ARRAY(9, m, temp_buffer);
        return rocsparse_status_success;
    }

    template <typename T>
    static rocsparse_status xcsrilu0(rocsparse_handle          handle,
                                     rocsparse_int             m,
                                     rocsparse_int             nnz,
                                     const rocsparse_mat_descr descr,
                                     T*                        csr_val,
                                     const rocsparse_int*      csr_row_ptr,
                                     const rocsparse_int*      csr_col_ind,
                                     rocsparse_mat_info        info,
                                     rocsparse_solve_policy    policy,
                                     void*                     temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::xcsrilu0_checkarg(
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer));

        auto csrilu0_info = (info != nullptr) ? info->get_csrilu0_info() : nullptr;

        _rocsparse_spmat_descr csr(rocsparse_format_csr,
                                   true,
                                   static_cast<int64_t>(1),
                                   m,
                                   m,
                                   nnz,
                                   rocsparse::get_datatype<T>(),
                                   csr_val,
                                   csr_val,
                                   static_cast<int64_t>(0),
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

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrilu0(handle,
                                                     &csr,
                                                     policy,
                                                     csrilu0_info,
                                                     info->get_boost(),
                                                     std::numeric_limits<size_t>::max(),
                                                     temp_buffer));
        return rocsparse_status_success;
    }
}

#define CIMPL(NAME, T)                                                                             \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,                             \
                                     rocsparse_int             m,                                  \
                                     rocsparse_int             nnz,                                \
                                     const rocsparse_mat_descr descr,                              \
                                     T*                        csr_val,                            \
                                     const rocsparse_int*      csr_row_ptr,                        \
                                     const rocsparse_int*      csr_col_ind,                        \
                                     rocsparse_mat_info        info,                               \
                                     rocsparse_solve_policy    policy,                             \
                                     void*                     temp_buffer)                        \
    try                                                                                            \
    {                                                                                              \
        ROCSPARSE_ROUTINE_TRACE;                                                                   \
                                                                                                   \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::xcsrilu0<T>(                                          \
            handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info, policy, temp_buffer)); \
        return rocsparse_status_success;                                                           \
    }                                                                                              \
    catch(...)                                                                                     \
    {                                                                                              \
        RETURN_ROCSPARSE_EXCEPTION();                                                              \
    }

CIMPL(rocsparse_dcsrilu0, double);
CIMPL(rocsparse_zcsrilu0, rocsparse_double_complex);
CIMPL(rocsparse_scsrilu0, float);
CIMPL(rocsparse_ccsrilu0, rocsparse_float_complex);

#undef CIMPL

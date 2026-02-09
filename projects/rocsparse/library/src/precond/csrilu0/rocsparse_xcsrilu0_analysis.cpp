/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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
    static rocsparse_status xcsrilu0_analysis_checkarg(rocsparse_handle          handle, //0
                                                       int64_t                   m, //1
                                                       int64_t                   nnz, //2
                                                       const rocsparse_mat_descr descr, //3
                                                       const void*               csr_val, //4
                                                       const void*               csr_row_ptr, //5
                                                       const void*               csr_col_ind, //6
                                                       rocsparse_mat_info        info, //7
                                                       rocsparse_analysis_policy analysis, //8
                                                       rocsparse_solve_policy    solve, //9
                                                       void*                     temp_buffer) //10
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
        ROCSPARSE_CHECKARG_ENUM(8, analysis);
        ROCSPARSE_CHECKARG_ENUM(9, solve);
        ROCSPARSE_CHECKARG_ARRAY(10, m, temp_buffer);
        ROCSPARSE_CHECKARG(
            9, solve, (solve != rocsparse_solve_policy_auto), rocsparse_status_invalid_value);

        return rocsparse_status_success;
    }

    template <typename T>
    rocsparse_status xcsrilu0_analysis(rocsparse_handle          handle,
                                       rocsparse_int             m,
                                       rocsparse_int             nnz,
                                       const rocsparse_mat_descr descr,
                                       const T*                  csr_val,
                                       const rocsparse_int*      csr_row_ptr,
                                       const rocsparse_int*      csr_col_ind,
                                       rocsparse_mat_info        info,
                                       rocsparse_analysis_policy analysis,
                                       rocsparse_solve_policy    solve,
                                       void*                     temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::xcsrilu0_analysis_checkarg(handle,
                                                                        m,
                                                                        nnz,
                                                                        descr,
                                                                        csr_val,
                                                                        csr_row_ptr,
                                                                        csr_col_ind,
                                                                        info,
                                                                        analysis,
                                                                        solve,
                                                                        temp_buffer));

        rocsparse_csrilu0_info csrilu0_info
            = (info != nullptr) ? info->get_csrilu0_info() : nullptr;

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

        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::csrilu0_analysis(handle, &csr, analysis, solve, &csrilu0_info, temp_buffer));
        return rocsparse_status_success;
    }
}

extern "C" rocsparse_status rocsparse_scsrilu0_analysis(rocsparse_handle          handle,
                                                        rocsparse_int             m,
                                                        rocsparse_int             nnz,
                                                        const rocsparse_mat_descr descr,
                                                        const float*              csr_val,
                                                        const rocsparse_int*      csr_row_ptr,
                                                        const rocsparse_int*      csr_col_ind,
                                                        rocsparse_mat_info        info,
                                                        rocsparse_analysis_policy analysis,
                                                        rocsparse_solve_policy    solve,
                                                        void*                     temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::xcsrilu0_analysis(handle,
                                                           m,
                                                           nnz,
                                                           descr,
                                                           csr_val,
                                                           csr_row_ptr,
                                                           csr_col_ind,
                                                           info,
                                                           analysis,
                                                           solve,
                                                           temp_buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_ccsrilu0_analysis(rocsparse_handle               handle,
                                                        rocsparse_int                  m,
                                                        rocsparse_int                  nnz,
                                                        const rocsparse_mat_descr      descr,
                                                        const rocsparse_float_complex* csr_val,
                                                        const rocsparse_int*           csr_row_ptr,
                                                        const rocsparse_int*           csr_col_ind,
                                                        rocsparse_mat_info             info,
                                                        rocsparse_analysis_policy      analysis,
                                                        rocsparse_solve_policy         solve,
                                                        void*                          temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::xcsrilu0_analysis(handle,
                                                           m,
                                                           nnz,
                                                           descr,
                                                           csr_val,
                                                           csr_row_ptr,
                                                           csr_col_ind,
                                                           info,
                                                           analysis,
                                                           solve,
                                                           temp_buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_dcsrilu0_analysis(rocsparse_handle          handle,
                                                        rocsparse_int             m,
                                                        rocsparse_int             nnz,
                                                        const rocsparse_mat_descr descr,
                                                        const double*             csr_val,
                                                        const rocsparse_int*      csr_row_ptr,
                                                        const rocsparse_int*      csr_col_ind,
                                                        rocsparse_mat_info        info,
                                                        rocsparse_analysis_policy analysis,
                                                        rocsparse_solve_policy    solve,
                                                        void*                     temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::xcsrilu0_analysis(handle,
                                                           m,
                                                           nnz,
                                                           descr,
                                                           csr_val,
                                                           csr_row_ptr,
                                                           csr_col_ind,
                                                           info,
                                                           analysis,
                                                           solve,
                                                           temp_buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_zcsrilu0_analysis(rocsparse_handle                handle,
                                                        rocsparse_int                   m,
                                                        rocsparse_int                   nnz,
                                                        const rocsparse_mat_descr       descr,
                                                        const rocsparse_double_complex* csr_val,
                                                        const rocsparse_int*            csr_row_ptr,
                                                        const rocsparse_int*            csr_col_ind,
                                                        rocsparse_mat_info              info,
                                                        rocsparse_analysis_policy       analysis,
                                                        rocsparse_solve_policy          solve,
                                                        void*                           temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::xcsrilu0_analysis(handle,
                                                           m,
                                                           nnz,
                                                           descr,
                                                           csr_val,
                                                           csr_row_ptr,
                                                           csr_col_ind,
                                                           info,
                                                           analysis,
                                                           solve,
                                                           temp_buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

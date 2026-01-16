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
#include "internal/level2/rocsparse_csrsv.h"
#include "rocsparse_control.hpp"
#include "rocsparse_csrsv.hpp"
#include "rocsparse_utility.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
namespace rocsparse
{
    static rocsparse_status xcsrsv_analysis_checkarg(rocsparse_handle          handle,
                                                     rocsparse_operation       trans,
                                                     rocsparse_int             m,
                                                     rocsparse_int             nnz,
                                                     const rocsparse_mat_descr descr,
                                                     const void*               csr_val,
                                                     const void*               csr_row_ptr,
                                                     const void*               csr_col_ind,
                                                     rocsparse_mat_info        info,
                                                     rocsparse_analysis_policy analysis,
                                                     rocsparse_solve_policy    solve,
                                                     void*                     temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;
        // Check for valid handle and matrix descriptor
        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_ENUM(1, trans);
        ROCSPARSE_CHECKARG_SIZE(2, m);
        ROCSPARSE_CHECKARG_SIZE(3, nnz);
        ROCSPARSE_CHECKARG_POINTER(4, descr);
        ROCSPARSE_CHECKARG_ARRAY(5, nnz, csr_val);
        ROCSPARSE_CHECKARG_ARRAY(6, m, csr_row_ptr);
        ROCSPARSE_CHECKARG_ARRAY(7, nnz, csr_col_ind);
        ROCSPARSE_CHECKARG_POINTER(8, info);
        ROCSPARSE_CHECKARG_ENUM(9, analysis);
        ROCSPARSE_CHECKARG_ENUM(10, solve);
        const rocsparse_int accept_temp_buffer = m;
        ROCSPARSE_CHECKARG_ARRAY(11, accept_temp_buffer, temp_buffer);
        return rocsparse_status_continue;
    }

    template <typename T>
    static rocsparse_status xcsrsv_analysis(rocsparse_handle          handle,
                                            rocsparse_operation       trans,
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

        if(m == 0)
            return rocsparse_status_success;
        rocsparse_status status = rocsparse::xcsrsv_analysis_checkarg(handle,
                                                                      trans,
                                                                      m,
                                                                      nnz,
                                                                      descr,
                                                                      csr_val,
                                                                      csr_row_ptr,
                                                                      csr_col_ind,
                                                                      info,
                                                                      analysis,
                                                                      solve,
                                                                      temp_buffer);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        rocsparse_csrsv_info   csrsv_info = (info != nullptr) ? info->get_csrsv_info() : nullptr;
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
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsv_analysis(
            handle, trans, &csr, analysis, solve, &csrsv_info, temp_buffer));

        return rocsparse_status_success;
    }

}
/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
#define CIMPL(NAME, T)                                                        \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,        \
                                     rocsparse_operation       trans,         \
                                     rocsparse_int             m,             \
                                     rocsparse_int             nnz,           \
                                     const rocsparse_mat_descr descr,         \
                                     const T*                  csr_val,       \
                                     const rocsparse_int*      csr_row_ptr,   \
                                     const rocsparse_int*      csr_col_ind,   \
                                     rocsparse_mat_info        info,          \
                                     rocsparse_analysis_policy analysis,      \
                                     rocsparse_solve_policy    solve,         \
                                     void*                     temp_buffer)   \
    try                                                                       \
    {                                                                         \
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::xcsrsv_analysis(handle,         \
                                                              trans,          \
                                                              m,              \
                                                              nnz,            \
                                                              descr,          \
                                                              csr_val,        \
                                                              csr_row_ptr,    \
                                                              csr_col_ind,    \
                                                              info,           \
                                                              analysis,       \
                                                              solve,          \
                                                              temp_buffer))); \
        return rocsparse_status_success;                                      \
    }                                                                         \
    catch(...)                                                                \
    {                                                                         \
        RETURN_ROCSPARSE_EXCEPTION();                                         \
    }

CIMPL(rocsparse_scsrsv_analysis, float);
CIMPL(rocsparse_dcsrsv_analysis, double);
CIMPL(rocsparse_ccsrsv_analysis, rocsparse_float_complex);
CIMPL(rocsparse_zcsrsv_analysis, rocsparse_double_complex);

#undef CIMPL

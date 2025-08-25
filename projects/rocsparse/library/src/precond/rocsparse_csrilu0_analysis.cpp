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
#include "internal/level2/rocsparse_csrsv.h"
#include "internal/precond/rocsparse_csrilu0.h"
#include "rocsparse_csrilu0.hpp"

namespace rocsparse
{

    template <typename T>
    rocsparse_status csrilu0_analysis_core(rocsparse_handle          handle,
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

        if(analysis == rocsparse_analysis_policy_reuse)
        {
            auto trm = info->get_csrilu0_info(rocsparse_operation_none, rocsparse_fill_mode_lower);
            if(trm == nullptr)
            {
                trm = (trm != nullptr) ? trm
                                       : info->get_csric0_info(rocsparse_operation_none,
                                                               rocsparse_fill_mode_lower);
                trm = (trm != nullptr) ? trm
                                       : info->get_csrsv_info(rocsparse_operation_none,
                                                              rocsparse_fill_mode_lower);
                trm = (trm != nullptr) ? trm
                                       : info->get_csrsm_info(rocsparse_operation_none,
                                                              rocsparse_fill_mode_lower);
                trm = (trm != nullptr) ? trm
                                       : info->get_csrsv_info(rocsparse_operation_transpose,
                                                              rocsparse_fill_mode_upper);
                trm = (trm != nullptr) ? trm
                                       : info->get_csrsm_info(rocsparse_operation_transpose,
                                                              rocsparse_fill_mode_upper);

                if(trm != nullptr)
                {
                    info->set_csrilu0_info(
                        rocsparse_operation_none, rocsparse_fill_mode_lower, trm);
                }
            }

            if(trm != nullptr)
            {
                return rocsparse_status_success;
            }
        }

        auto csrilu0_info = info->get_csrilu0_info();
        // Perform analysis
        RETURN_IF_ROCSPARSE_ERROR(csrilu0_info->recreate(rocsparse_operation_none,
                                                         rocsparse_fill_mode_lower,
                                                         handle,
                                                         rocsparse_operation_none,
                                                         m,
                                                         nnz,
                                                         descr,
                                                         csr_val,
                                                         csr_row_ptr,
                                                         csr_col_ind,
                                                         temp_buffer));

        // setup info->singular_pivot
        csrilu0_info->create_singular_pivot_async(rocsparse::get_indextype<rocsparse_int>(),
                                                  handle->stream);

        RETURN_IF_HIP_ERROR(hipMemcpyAsync(csrilu0_info->get_singular_pivot(),
                                           csrilu0_info->get_zero_pivot(),
                                           sizeof(rocsparse_int),
                                           hipMemcpyDeviceToDevice,
                                           handle->stream));

        return rocsparse_status_success;
    }

    static rocsparse_status csrilu0_analysis_quickreturn(rocsparse_handle          handle,
                                                         int64_t                   m,
                                                         int64_t                   nnz,
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

        if(m == 0)
        {
            return rocsparse_status_success;
        }

        return rocsparse_status_continue;
    }

    static rocsparse_status csrilu0_analysis_checkarg(rocsparse_handle          handle, //0
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

        const rocsparse_status status = rocsparse::csrilu0_analysis_quickreturn(handle,
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
        return rocsparse_status_continue;
    }

    template <typename T>
    rocsparse_status csrilu0_analysis_impl(rocsparse_handle          handle,
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

        // Logging
        rocsparse::log_trace(handle,
                             rocsparse::replaceX<T>("rocsparse_Xcsrilu0_analysis"),
                             m,
                             nnz,
                             (const void*&)descr,
                             (const void*&)csr_val,
                             (const void*&)csr_row_ptr,
                             (const void*&)csr_col_ind,
                             (const void*&)info,
                             solve,
                             analysis);

        const rocsparse_status status = rocsparse::csrilu0_analysis_checkarg(handle,
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

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrilu0_analysis_core(handle,
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

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrilu0_analysis_impl(handle,
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

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrilu0_analysis_impl(handle,
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

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrilu0_analysis_impl(handle,
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

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrilu0_analysis_impl(handle,
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

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
#include "internal/level3/rocsparse_csrsm.h"
#include "rocsparse_csrsm.hpp"

#include "../level2/rocsparse_csrsv.hpp"
#include "common.h"
#include "control.h"
#include "utility.h"

template <typename I, typename J, typename T>
rocsparse_status rocsparse::csrsm_analysis_core(rocsparse_handle          handle,
                                                rocsparse_operation       trans_A,
                                                rocsparse_operation       trans_B,
                                                J                         m,
                                                J                         nrhs,
                                                I                         nnz,
                                                const T*                  alpha,
                                                const rocsparse_mat_descr descr,
                                                const T*                  csr_val,
                                                const I*                  csr_row_ptr,
                                                const J*                  csr_col_ind,
                                                const T*                  B,
                                                int64_t                   ldb,
                                                rocsparse_mat_info        info,
                                                rocsparse_analysis_policy analysis,
                                                rocsparse_solve_policy    solve,
                                                void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(nrhs == 1)
    {
        //
        // Call csrsv.
        //
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsv_analysis_template(handle,
                                                                     trans_A,
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

    // Switch between lower and upper triangular analysis
    if(descr->fill_mode == rocsparse_fill_mode_upper)
    {
        // Differentiate the analysis policies
        if(analysis == rocsparse_analysis_policy_reuse)
        {
            // We try to re-use already analyzed upper part, if available.
            // It is the user's responsibility that this data is still valid,
            // since he passed the 'reuse' flag.

            // If csrsm meta data is already available, do nothing
            if(trans_A == rocsparse_operation_none && info->csrsm_upper_info != nullptr)
            {
                return rocsparse_status_success;
            }
            else if(trans_A == rocsparse_operation_transpose && info->csrsmt_upper_info != nullptr)
            {
                return rocsparse_status_success;
            }
            else if(trans_A == rocsparse_operation_conjugate_transpose
                    && info->csrsmt_upper_info != nullptr)
            {
                return rocsparse_status_success;
            }

            // Check for other upper analysis meta data

            if(trans_A == rocsparse_operation_none && info->csrsv_upper_info != nullptr)
            {
                // csrsv meta data
                info->csrsm_upper_info = info->csrsv_upper_info;
                return rocsparse_status_success;
            }

            if(trans_A == rocsparse_operation_transpose && info->csrsvt_upper_info != nullptr)
            {
                // csrsv meta data
                info->csrsmt_upper_info = info->csrsvt_upper_info;
                return rocsparse_status_success;
            }

            if(trans_A == rocsparse_operation_conjugate_transpose
               && info->csrsvt_upper_info != nullptr)
            {
                // csrsv meta data
                info->csrsmt_upper_info = info->csrsvt_upper_info;
                return rocsparse_status_success;
            }
        }

        // User is explicitly asking to force a re-analysis, or no valid data has been
        // found to be re-used

        // Clear csrsm info
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::destroy_trm_info((trans_A == rocsparse_operation_none)
                                                                  ? info->csrsm_upper_info
                                                                  : info->csrsmt_upper_info));

        // Create csrsm info
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::create_trm_info((trans_A == rocsparse_operation_none)
                                                                 ? &info->csrsm_upper_info
                                                                 : &info->csrsmt_upper_info));

        // Perform analysis
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::trm_analysis(handle,
                                                          trans_A,
                                                          m,
                                                          nnz,
                                                          descr,
                                                          csr_val,
                                                          csr_row_ptr,
                                                          csr_col_ind,
                                                          (trans_A == rocsparse_operation_none)
                                                              ? info->csrsm_upper_info
                                                              : info->csrsmt_upper_info,
                                                          (J**)&info->zero_pivot,
                                                          temp_buffer));
    }
    else
    {
        // Differentiate the analysis policies
        if(analysis == rocsparse_analysis_policy_reuse)
        {
            // We try to re-use already analyzed lower part, if available.
            // It is the user's responsibility that this data is still valid,
            // since he passed the 'reuse' flag.

            // If csrsm meta data is already available, do nothing
            if(trans_A == rocsparse_operation_none && info->csrsm_lower_info != nullptr)
            {
                return rocsparse_status_success;
            }
            else if(trans_A == rocsparse_operation_transpose && info->csrsmt_lower_info != nullptr)
            {
                return rocsparse_status_success;
            }
            else if(trans_A == rocsparse_operation_conjugate_transpose
                    && info->csrsmt_lower_info != nullptr)
            {
                return rocsparse_status_success;
            }

            // Check for other lower analysis meta data

            if(trans_A == rocsparse_operation_none && info->csrilu0_info != nullptr)
            {
                // csrilu0 meta data
                info->csrsm_lower_info = info->csrilu0_info;
                return rocsparse_status_success;
            }
            else if(trans_A == rocsparse_operation_none && info->csric0_info != nullptr)
            {
                // csric0 meta data
                info->csrsm_lower_info = info->csric0_info;
                return rocsparse_status_success;
            }
            else if(trans_A == rocsparse_operation_none && info->csrsv_lower_info != nullptr)
            {
                // csrsv meta data
                info->csrsm_lower_info = info->csrsv_lower_info;
                return rocsparse_status_success;
            }

            if(trans_A == rocsparse_operation_transpose && info->csrsvt_lower_info != nullptr)
            {
                // csrsv meta data
                info->csrsm_upper_info = info->csrsvt_lower_info;
                return rocsparse_status_success;
            }

            if(trans_A == rocsparse_operation_conjugate_transpose
               && info->csrsvt_lower_info != nullptr)
            {
                // csrsv meta data
                info->csrsm_upper_info = info->csrsvt_lower_info;
                return rocsparse_status_success;
            }
        }

        // User is explicitly asking to force a re-analysis, or no valid data has been
        // found to be re-used

        // Clear csrsm info
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::destroy_trm_info((trans_A == rocsparse_operation_none)
                                                                  ? info->csrsm_lower_info
                                                                  : info->csrsmt_lower_info));

        // Create csrsm info
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::create_trm_info((trans_A == rocsparse_operation_none)
                                                                 ? &info->csrsm_lower_info
                                                                 : &info->csrsmt_lower_info));

        // Perform analysis
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::trm_analysis(handle,
                                                          trans_A,
                                                          m,
                                                          nnz,
                                                          descr,
                                                          csr_val,
                                                          csr_row_ptr,
                                                          csr_col_ind,
                                                          (trans_A == rocsparse_operation_none)
                                                              ? info->csrsm_lower_info
                                                              : info->csrsmt_lower_info,
                                                          (J**)&info->zero_pivot,
                                                          temp_buffer));
    }

    return rocsparse_status_success;
}

rocsparse_status rocsparse::csrsm_analysis_quickreturn(rocsparse_handle          handle,
                                                       rocsparse_operation       trans_A,
                                                       rocsparse_operation       trans_B,
                                                       int64_t                   m,
                                                       int64_t                   nrhs,
                                                       int64_t                   nnz,
                                                       const void*               alpha,
                                                       const rocsparse_mat_descr descr,
                                                       const void*               csr_val,
                                                       const void*               csr_row_ptr,
                                                       const void*               csr_col_ind,
                                                       const void*               B,
                                                       int64_t                   ldb,
                                                       rocsparse_mat_info        info,
                                                       rocsparse_analysis_policy analysis,
                                                       rocsparse_solve_policy    solve,
                                                       void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(m == 0 || nrhs == 0)
    {
        return rocsparse_status_success;
    }

    return rocsparse_status_continue;
}

namespace rocsparse
{
    static rocsparse_status csrsm_analysis_checkarg(rocsparse_handle          handle, //0
                                                    rocsparse_operation       trans_A, //1
                                                    rocsparse_operation       trans_B, //2
                                                    int64_t                   m, //3
                                                    int64_t                   nrhs, //4
                                                    int64_t                   nnz, //5
                                                    const void*               alpha, //6
                                                    const rocsparse_mat_descr descr, //7
                                                    const void*               csr_val, //8
                                                    const void*               csr_row_ptr, //9
                                                    const void*               csr_col_ind, //10
                                                    const void*               B, //11
                                                    int64_t                   ldb, //12
                                                    rocsparse_mat_info        info, //13
                                                    rocsparse_analysis_policy analysis, //14
                                                    rocsparse_solve_policy    solve, //15
                                                    void*                     temp_buffer) //16
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_ENUM(1, trans_A);
        ROCSPARSE_CHECKARG_ENUM(2, trans_B);
        ROCSPARSE_CHECKARG_SIZE(3, m);
        ROCSPARSE_CHECKARG_SIZE(4, nrhs);
        ROCSPARSE_CHECKARG_SIZE(5, nnz);
        ROCSPARSE_CHECKARG(12,
                           ldb,
                           (trans_B == rocsparse_operation_none && ldb < m),
                           rocsparse_status_invalid_size);
        ROCSPARSE_CHECKARG(12,
                           ldb,
                           ((trans_B == rocsparse_operation_transpose
                             || trans_B == rocsparse_operation_conjugate_transpose)
                            && ldb < nrhs),
                           rocsparse_status_invalid_size);

        const rocsparse_status status = rocsparse::csrsm_analysis_quickreturn(handle,
                                                                              trans_A,
                                                                              trans_B,
                                                                              m,
                                                                              nrhs,
                                                                              nnz,
                                                                              alpha,
                                                                              descr,
                                                                              csr_val,
                                                                              csr_row_ptr,
                                                                              csr_col_ind,
                                                                              B,
                                                                              ldb,
                                                                              info,
                                                                              analysis,
                                                                              solve,
                                                                              temp_buffer);

        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG_POINTER(7, descr);

        ROCSPARSE_CHECKARG(7,
                           descr,
                           (descr->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(7,
                           descr,
                           (descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);

        ROCSPARSE_CHECKARG_ARRAY(8, nnz, csr_val);
        ROCSPARSE_CHECKARG_ARRAY(9, m, csr_row_ptr);
        ROCSPARSE_CHECKARG_ARRAY(10, nnz, csr_col_ind);
        ROCSPARSE_CHECKARG_POINTER(13, info);
        ROCSPARSE_CHECKARG_ENUM(14, analysis);
        ROCSPARSE_CHECKARG_ENUM(15, solve);

        ROCSPARSE_CHECKARG_POINTER(6, alpha);
        ROCSPARSE_CHECKARG_POINTER(11, B);
        ROCSPARSE_CHECKARG_POINTER(16, temp_buffer);
        return rocsparse_status_continue;
    }

    template <typename I, typename J, typename T>
    static rocsparse_status csrsm_analysis_impl(rocsparse_handle          handle,
                                                rocsparse_operation       trans_A,
                                                rocsparse_operation       trans_B,
                                                J                         m,
                                                J                         nrhs,
                                                I                         nnz,
                                                const T*                  alpha,
                                                const rocsparse_mat_descr descr,
                                                const T*                  csr_val,
                                                const I*                  csr_row_ptr,
                                                const J*                  csr_col_ind,
                                                const T*                  B,
                                                int64_t                   ldb,
                                                rocsparse_mat_info        info,
                                                rocsparse_analysis_policy analysis,
                                                rocsparse_solve_policy    solve,
                                                void*                     temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Logging
        rocsparse::log_trace(handle,
                             rocsparse::replaceX<T>("rocsparse_Xcsrsm_analysis"),
                             trans_A,
                             trans_B,
                             m,
                             nrhs,
                             nnz,
                             LOG_TRACE_SCALAR_VALUE(handle, alpha),
                             (const void*&)descr,
                             (const void*&)csr_val,
                             (const void*&)csr_row_ptr,
                             (const void*&)csr_col_ind,
                             (const void*&)B,
                             ldb,
                             (const void*&)info,
                             analysis,
                             solve,
                             (const void*&)temp_buffer);

        const rocsparse_status status = rocsparse::csrsm_analysis_checkarg(handle,
                                                                           trans_A,
                                                                           trans_B,
                                                                           m,
                                                                           nrhs,
                                                                           nnz,
                                                                           alpha,
                                                                           descr,
                                                                           csr_val,
                                                                           csr_row_ptr,
                                                                           csr_col_ind,
                                                                           B,
                                                                           ldb,
                                                                           info,
                                                                           analysis,
                                                                           solve,
                                                                           temp_buffer);

        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_analysis_core(handle,
                                                                 trans_A,
                                                                 trans_B,
                                                                 m,
                                                                 nrhs,
                                                                 nnz,
                                                                 alpha,
                                                                 descr,
                                                                 csr_val,
                                                                 csr_row_ptr,
                                                                 csr_col_ind,
                                                                 B,
                                                                 ldb,
                                                                 info,
                                                                 analysis,
                                                                 solve,
                                                                 temp_buffer));
        return rocsparse_status_success;
    }
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                            \
    template rocsparse_status rocsparse::csrsm_analysis_core(rocsparse_handle          handle,      \
                                                             rocsparse_operation       trans_A,     \
                                                             rocsparse_operation       trans_B,     \
                                                             JTYPE                     m,           \
                                                             JTYPE                     nrhs,        \
                                                             ITYPE                     nnz,         \
                                                             const TTYPE*              alpha,       \
                                                             const rocsparse_mat_descr descr,       \
                                                             const TTYPE*              csr_val,     \
                                                             const ITYPE*              csr_row_ptr, \
                                                             const JTYPE*              csr_col_ind, \
                                                             const TTYPE*              B,           \
                                                             int64_t                   ldb,         \
                                                             rocsparse_mat_info        info,        \
                                                             rocsparse_analysis_policy analysis,    \
                                                             rocsparse_solve_policy    solve,       \
                                                             void*                     temp_buffer);

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);
#undef INSTANTIATE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
#define C_IMPL(NAME, TTYPE)                                                     \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,          \
                                     rocsparse_operation       trans_A,         \
                                     rocsparse_operation       trans_B,         \
                                     rocsparse_int             m,               \
                                     rocsparse_int             nrhs,            \
                                     rocsparse_int             nnz,             \
                                     const TTYPE*              alpha,           \
                                     const rocsparse_mat_descr descr,           \
                                     const TTYPE*              csr_val,         \
                                     const rocsparse_int*      csr_row_ptr,     \
                                     const rocsparse_int*      csr_col_ind,     \
                                     const TTYPE*              B,               \
                                     rocsparse_int             ldb,             \
                                     rocsparse_mat_info        info,            \
                                     rocsparse_analysis_policy analysis,        \
                                     rocsparse_solve_policy    solve,           \
                                     void*                     temp_buffer)     \
    try                                                                         \
    {                                                                           \
        ROCSPARSE_ROUTINE_TRACE;                                                \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_analysis_impl(handle,        \
                                                                 trans_A,       \
                                                                 trans_B,       \
                                                                 m,             \
                                                                 nrhs,          \
                                                                 nnz,           \
                                                                 alpha,         \
                                                                 descr,         \
                                                                 csr_val,       \
                                                                 csr_row_ptr,   \
                                                                 csr_col_ind,   \
                                                                 B,             \
                                                                 ldb,           \
                                                                 info,          \
                                                                 analysis,      \
                                                                 solve,         \
                                                                 temp_buffer)); \
        return rocsparse_status_success;                                        \
    }                                                                           \
    catch(...)                                                                  \
    {                                                                           \
        RETURN_ROCSPARSE_EXCEPTION();                                           \
    }

C_IMPL(rocsparse_scsrsm_analysis, float);
C_IMPL(rocsparse_dcsrsm_analysis, double);
C_IMPL(rocsparse_ccsrsm_analysis, rocsparse_float_complex);
C_IMPL(rocsparse_zcsrsm_analysis, rocsparse_double_complex);

#undef C_IMPL

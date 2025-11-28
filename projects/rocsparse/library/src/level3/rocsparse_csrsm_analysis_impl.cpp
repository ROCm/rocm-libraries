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

#include "internal/level3/rocsparse_csrsm.h"
#include "rocsparse_csrsm.hpp"

#include "../level2/rocsparse_csrsv.hpp"
#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

template <typename I, typename J, typename T>
rocsparse_status rocsparse::csrsm_analysis_core(rocsparse_handle          handle,
                                                rocsparse_operation       trans_A,
                                                rocsparse_operation       trans_B,
                                                int64_t                   m_,
                                                int64_t                   nrhs_,
                                                int64_t                   nnz_,
                                                const void*               alpha_,
                                                const rocsparse_mat_descr descr,
                                                const void*               csr_val_,
                                                const void*               csr_row_ptr_,
                                                const void*               csr_col_ind_,
                                                const void*               B_,
                                                int64_t                   ldb,
                                                rocsparse_mat_info        info,
                                                rocsparse_analysis_policy analysis,
                                                rocsparse_solve_policy    solve,
                                                rocsparse_csrsm_info*     p_csrsm_info,
                                                void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    const J  m           = static_cast<J>(m_);
    const J  nrhs        = static_cast<J>(nrhs_);
    const I  nnz         = static_cast<I>(nnz_);
    const T* csr_val     = reinterpret_cast<const T*>(csr_val_);
    const I* csr_row_ptr = reinterpret_cast<const I*>(csr_row_ptr_);
    const J* csr_col_ind = reinterpret_cast<const J*>(csr_col_ind_);

    if(nrhs == 1)
    {
        //
        // Call csrsv.
        //
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrsv_analysis_template<I, J, T>(handle,
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
                                                                               p_csrsm_info,
                                                                               temp_buffer)));
        return rocsparse_status_success;
    }

    auto csrsm_info = p_csrsm_info[0];

    // Differentiate the analysis policies
    if(analysis == rocsparse_analysis_policy_reuse)
    {
        //
        //
        //
        rocsparse::trm_info_t* p = nullptr;

        p = (p != nullptr) ? p : info->get_csrsm_info(trans_A, descr->fill_mode);

        if((descr->fill_mode == rocsparse_fill_mode_lower) && (trans_A == rocsparse_operation_none))
        {
            p = (p != nullptr) ? p : info->get_csrilu0_info(trans_A, descr->fill_mode);
            p = (p != nullptr) ? p : info->get_csric0_info(trans_A, descr->fill_mode);
        }

        p = (p != nullptr) ? p : info->get_csrsv_info(trans_A, descr->fill_mode);
        if(p != nullptr)
        {
            info->set_csrsm_info(trans_A, descr->fill_mode, p);
            return rocsparse_status_success;
        }
    }

    if(csrsm_info == nullptr)
    {
        csrsm_info      = new _rocsparse_csrsm_info();
        p_csrsm_info[0] = csrsm_info;
    }

    // Perform analysis
    RETURN_IF_ROCSPARSE_ERROR(csrsm_info->recreate(
        handle, trans_A, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, temp_buffer));

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

        rocsparse_csrsm_info csrsm_info = info->get_csrsm_info();
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrsm_analysis_core<I, J, T>(handle,
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
                                                                           &csrsm_info,
                                                                           temp_buffer)));

        return rocsparse_status_success;
    }
}

#define INSTANTIATE(I, J, T)                                           \
    template rocsparse_status rocsparse::csrsm_analysis_core<I, J, T>( \
        rocsparse_handle          handle,                              \
        rocsparse_operation       trans_A,                             \
        rocsparse_operation       trans_B,                             \
        int64_t                   m,                                   \
        int64_t                   nrhs,                                \
        int64_t                   nnz,                                 \
        const void*               alpha,                               \
        const rocsparse_mat_descr descr,                               \
        const void*               csr_val,                             \
        const void*               csr_row_ptr,                         \
        const void*               csr_col_ind,                         \
        const void*               B,                                   \
        int64_t                   ldb,                                 \
        rocsparse_mat_info        info,                                \
        rocsparse_analysis_policy analysis,                            \
        rocsparse_solve_policy    solve,                               \
        rocsparse_csrsm_info*     p_csrsm_info,                        \
        void*                     temp_buffer)

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);

INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);

INSTANTIATE(int32_t, int64_t, float);
INSTANTIATE(int32_t, int64_t, double);
INSTANTIATE(int32_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int64_t, rocsparse_double_complex);

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
#define C_IMPL(NAME, T)                                                         \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,          \
                                     rocsparse_operation       trans_A,         \
                                     rocsparse_operation       trans_B,         \
                                     rocsparse_int             m,               \
                                     rocsparse_int             nrhs,            \
                                     rocsparse_int             nnz,             \
                                     const T*                  alpha,           \
                                     const rocsparse_mat_descr descr,           \
                                     const T*                  csr_val,         \
                                     const rocsparse_int*      csr_row_ptr,     \
                                     const rocsparse_int*      csr_col_ind,     \
                                     const T*                  B,               \
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

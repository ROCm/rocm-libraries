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

#include "internal/level3/rocsparse_csrsm.h"
#include "rocsparse_csrsm.hpp"

#include "../../level2/rocsparse_csrsv.hpp"
#include "rocsparse_csrsm.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <typename T>
    rocsparse_status xcsrsm_analysis(rocsparse_handle          handle,
                                     rocsparse_operation       trans_A,
                                     rocsparse_operation       trans_B,
                                     rocsparse_int             m,
                                     rocsparse_int             nrhs,
                                     rocsparse_int             nnz,
                                     const void*               alpha,
                                     const rocsparse_mat_descr descr,
                                     const void*               csr_val,
                                     const rocsparse_int*      csr_row_ptr,
                                     const rocsparse_int*      csr_col_ind,
                                     const void*               B,
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
                             LOG_TRACE_SCALAR_VALUE(handle, reinterpret_cast<const T*>(alpha)),
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

        if(m == 0 || nrhs == 0)
        {
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

        rocsparse_error* p_error = nullptr;

        _rocsparse_spmat_descr local_A(rocsparse_format_csr,

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

        const int64_t          B_m = (trans_B == rocsparse_operation_none) ? m : nrhs;
        const int64_t          B_n = (trans_B == rocsparse_operation_none) ? nrhs : m;
        _rocsparse_dnmat_descr local_B{true,
                                       B_m,
                                       B_n,
                                       ldb,
                                       nullptr,
                                       B,
                                       rocsparse::get_datatype<T>(),
                                       rocsparse_order_row, // <----- it's not used !
                                       1,
                                       0};

        _rocsparse_dnvec_descr local_alpha(
            1, 1, rocsparse::get_datatype<T>(), alpha, nullptr, 1, 0, handle->pointer_mode);

        rocsparse_csrsm_info csrsm_info = (info != nullptr) ? info->get_csrsm_info() : nullptr;

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_analysis(handle,
                                                            nrhs,
                                                            trans_A,
                                                            trans_B,
                                                            &local_alpha,
                                                            &local_A,
                                                            &local_B,
                                                            analysis,
                                                            &csrsm_info,
                                                            std::numeric_limits<size_t>::max(),
                                                            temp_buffer,
                                                            p_error));

        return rocsparse_status_success;
    }
}

/*
   * ===========================================================================
   *    C wrapper
   * ===========================================================================
   */
#define C_IMPL(NAME, T)                                                        \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,         \
                                     rocsparse_operation       trans_A,        \
                                     rocsparse_operation       trans_B,        \
                                     rocsparse_int             m,              \
                                     rocsparse_int             nrhs,           \
                                     rocsparse_int             nnz,            \
                                     const T*                  alpha,          \
                                     const rocsparse_mat_descr descr,          \
                                     const T*                  csr_val,        \
                                     const rocsparse_int*      csr_row_ptr,    \
                                     const rocsparse_int*      csr_col_ind,    \
                                     const T*                  B,              \
                                     rocsparse_int             ldb,            \
                                     rocsparse_mat_info        info,           \
                                     rocsparse_analysis_policy analysis,       \
                                     rocsparse_solve_policy    solve,          \
                                     void*                     temp_buffer)    \
    try                                                                        \
    {                                                                          \
        ROCSPARSE_ROUTINE_TRACE;                                               \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::xcsrsm_analysis<T>(handle,        \
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
        return rocsparse_status_success;                                       \
    }                                                                          \
    catch(...)                                                                 \
    {                                                                          \
        RETURN_ROCSPARSE_EXCEPTION();                                          \
    }

C_IMPL(rocsparse_scsrsm_analysis, float);
C_IMPL(rocsparse_dcsrsm_analysis, double);
C_IMPL(rocsparse_ccsrsm_analysis, rocsparse_float_complex);
C_IMPL(rocsparse_zcsrsm_analysis, rocsparse_double_complex);

#undef C_IMPL

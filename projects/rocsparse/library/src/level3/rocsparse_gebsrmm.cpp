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

#include "internal/level3/rocsparse_gebsrmm.h"
#include "rocsparse_gebsrmm.hpp"

#include "../level2/rocsparse_gebsrmv.hpp"
#include "rocsparse_bsrmm.hpp"
#include "rocsparse_common.h"

#include "rocsparse_common.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <typename T>
    rocsparse_status gebsrmm_template_small(rocsparse_handle          handle,
                                            rocsparse_direction       dir,
                                            rocsparse_operation       trans_A,
                                            rocsparse_operation       trans_B,
                                            rocsparse_int             mb,
                                            rocsparse_int             n,
                                            rocsparse_int             kb,
                                            rocsparse_int             nnzb,
                                            const T*                  alpha,
                                            const rocsparse_mat_descr descr,
                                            const T*                  bsr_val,
                                            const rocsparse_int*      bsr_row_ptr,
                                            const rocsparse_int*      bsr_col_ind,
                                            rocsparse_int             row_block_dim,
                                            rocsparse_int             col_block_dim,
                                            const T*                  B,
                                            int64_t                   ldb,
                                            const T*                  beta,
                                            T*                        C,
                                            int64_t                   ldc);

    template <typename T>
    rocsparse_status gebsrmm_template_large_ext(rocsparse_handle          handle,
                                                rocsparse_direction       dir,
                                                rocsparse_operation       trans_A,
                                                rocsparse_operation       trans_B,
                                                rocsparse_int             mb,
                                                rocsparse_int             n,
                                                rocsparse_int             kb,
                                                rocsparse_int             nnzb,
                                                const T*                  alpha,
                                                const rocsparse_mat_descr descr,
                                                const T*                  bsr_val,
                                                const rocsparse_int*      bsr_row_ptr,
                                                const rocsparse_int*      bsr_col_ind,
                                                rocsparse_int             row_block_dim,
                                                rocsparse_int             col_block_dim,
                                                const T*                  B,
                                                int64_t                   ldb,
                                                const T*                  beta,
                                                T*                        C,
                                                int64_t                   ldc);

    template <typename T>
    rocsparse_status gebsrmm_template_general(rocsparse_handle          handle,
                                              rocsparse_direction       dir,
                                              rocsparse_operation       trans_A,
                                              rocsparse_operation       trans_B,
                                              rocsparse_int             mb,
                                              rocsparse_int             n,
                                              rocsparse_int             kb,
                                              rocsparse_int             nnzb,
                                              const T*                  alpha,
                                              const rocsparse_mat_descr descr,
                                              const T*                  bsr_val,
                                              const rocsparse_int*      bsr_row_ptr,
                                              const rocsparse_int*      bsr_col_ind,
                                              rocsparse_int             row_block_dim,
                                              rocsparse_int             col_block_dim,
                                              const T*                  B,
                                              int64_t                   ldb,
                                              const T*                  beta,
                                              T*                        C,
                                              int64_t                   ldc);
}

template <typename T>
rocsparse_status rocsparse::gebsrmm_template_dispatch(rocsparse_handle          handle,
                                                      rocsparse_direction       dir,
                                                      rocsparse_operation       trans_A,
                                                      rocsparse_operation       trans_B,
                                                      rocsparse_int             mb,
                                                      rocsparse_int             n,
                                                      rocsparse_int             kb,
                                                      rocsparse_int             nnzb,
                                                      const T*                  alpha,
                                                      const rocsparse_mat_descr descr,
                                                      const T*                  bsr_val,
                                                      const rocsparse_int*      bsr_row_ptr,
                                                      const rocsparse_int*      bsr_col_ind,
                                                      rocsparse_int             row_block_dim,
                                                      rocsparse_int             col_block_dim,
                                                      const T*                  B,
                                                      int64_t                   ldb,
                                                      const T*                  beta,
                                                      T*                        C,
                                                      int64_t                   ldc)
{
    ROCSPARSE_ROUTINE_TRACE;

    const rocsparse_int block_dim = rocsparse::max(row_block_dim, col_block_dim);
    if(row_block_dim == col_block_dim)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrmm_template_dispatch(handle,
                                                                     dir,
                                                                     trans_A,
                                                                     trans_B,
                                                                     rocsparse_bsrmm_alg_default,
                                                                     mb,
                                                                     n,
                                                                     kb,
                                                                     nnzb,
                                                                     static_cast<rocsparse_int>(1),
                                                                     static_cast<int64_t>(0),
                                                                     static_cast<int64_t>(0),
                                                                     alpha,
                                                                     descr,
                                                                     bsr_val,
                                                                     bsr_row_ptr,
                                                                     bsr_col_ind,
                                                                     block_dim,
                                                                     B,
                                                                     ldb,
                                                                     static_cast<rocsparse_int>(1),
                                                                     static_cast<int64_t>(0),
                                                                     rocsparse_order_column,
                                                                     beta,
                                                                     C,
                                                                     ldc,
                                                                     static_cast<rocsparse_int>(1),
                                                                     static_cast<int64_t>(0),
                                                                     rocsparse_order_column));
        return rocsparse_status_success;
    }

    // If n is only 1 and B are non-transposed, then call gebsrmv
    if(n == 1)
    {
        if(trans_B == rocsparse_operation_none)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::gebsrmv_template_dispatch(handle,
                                                                           dir,
                                                                           trans_A,
                                                                           mb,
                                                                           kb,
                                                                           nnzb,
                                                                           alpha,
                                                                           descr,
                                                                           bsr_val,
                                                                           bsr_row_ptr,
                                                                           bsr_col_ind,
                                                                           row_block_dim,
                                                                           col_block_dim,
                                                                           B,
                                                                           beta,
                                                                           C));
            return rocsparse_status_success;
        }
    }

    if(block_dim <= 4)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gebsrmm_template_small(handle,
                                                                    dir,
                                                                    trans_A,
                                                                    trans_B,
                                                                    mb,
                                                                    n,
                                                                    kb,
                                                                    nnzb,
                                                                    alpha,
                                                                    descr,
                                                                    bsr_val,
                                                                    bsr_row_ptr,
                                                                    bsr_col_ind,
                                                                    row_block_dim,
                                                                    col_block_dim,
                                                                    B,
                                                                    ldb,
                                                                    beta,
                                                                    C,
                                                                    ldc));
        return rocsparse_status_success;
    }
    else if(block_dim <= 32)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gebsrmm_template_large_ext(handle,
                                                                        dir,
                                                                        trans_A,
                                                                        trans_B,
                                                                        mb,
                                                                        n,
                                                                        kb,
                                                                        nnzb,
                                                                        alpha,
                                                                        descr,
                                                                        bsr_val,
                                                                        bsr_row_ptr,
                                                                        bsr_col_ind,
                                                                        row_block_dim,
                                                                        col_block_dim,
                                                                        B,
                                                                        ldb,
                                                                        beta,
                                                                        C,
                                                                        ldc));
        return rocsparse_status_success;
    }
    else
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gebsrmm_template_general(handle,
                                                                      dir,
                                                                      trans_A,
                                                                      trans_B,
                                                                      mb,
                                                                      n,
                                                                      kb,
                                                                      nnzb,
                                                                      alpha,
                                                                      descr,
                                                                      bsr_val,
                                                                      bsr_row_ptr,
                                                                      bsr_col_ind,
                                                                      row_block_dim,
                                                                      col_block_dim,
                                                                      B,
                                                                      ldb,
                                                                      beta,
                                                                      C,
                                                                      ldc));
        return rocsparse_status_success;
    }
}

namespace rocsparse
{
    template <typename T>
    static rocsparse_status gebsrmm_quickreturn(rocsparse_handle          handle,
                                                rocsparse_direction       dir,
                                                rocsparse_operation       trans_A,
                                                rocsparse_operation       trans_B,
                                                int64_t                   mb,
                                                int64_t                   n,
                                                int64_t                   kb,
                                                int64_t                   nnzb,
                                                const T*                  alpha,
                                                const rocsparse_mat_descr descr,
                                                const void*               bsr_val,
                                                const void*               bsr_row_ptr,
                                                const void*               bsr_col_ind,
                                                int64_t                   row_block_dim,
                                                int64_t                   col_block_dim,
                                                const void*               B,
                                                int64_t                   ldb,
                                                const T*                  beta,
                                                T*                        C,
                                                int64_t                   ldc)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Quick return if possible
        if(mb == 0 || n == 0 || kb == 0)
        {
            // matrix never accessed however still need to update C matrix
            const int64_t m      = row_block_dim * mb;
            const int64_t C_size = m * n;
            if(C_size > 0)
            {
                if(C == nullptr && beta == nullptr)
                {
                    return rocsparse_status_invalid_pointer;
                }

                RETURN_IF_ROCSPARSE_ERROR(rocsparse::scale_2d_array(
                    handle, m, n, ldc, 1, 0, beta, C, rocsparse_order_column));
            }

            return rocsparse_status_success;
        }
        return rocsparse_status_continue;
    }

    template <typename T>
    static rocsparse_status gebsrmm_checkarg(rocsparse_handle          handle, //0
                                             rocsparse_direction       dir, //1
                                             rocsparse_operation       trans_A, //2
                                             rocsparse_operation       trans_B, //3
                                             rocsparse_int             mb, //4
                                             rocsparse_int             n, //5
                                             rocsparse_int             kb, //6
                                             rocsparse_int             nnzb, //7
                                             const T*                  alpha, //8
                                             const rocsparse_mat_descr descr, //9
                                             const T*                  bsr_val, //10
                                             const rocsparse_int*      bsr_row_ptr, //11
                                             const rocsparse_int*      bsr_col_ind, //12
                                             rocsparse_int             row_block_dim, //13
                                             rocsparse_int             col_block_dim, //14
                                             const T*                  B, //15
                                             int64_t                   ldb, //16
                                             const T*                  beta, //17
                                             T*                        C, //18
                                             int64_t                   ldc) //19
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_POINTER(9, descr);
        ROCSPARSE_CHECKARG_ENUM(1, dir);
        ROCSPARSE_CHECKARG_ENUM(2, trans_A);
        ROCSPARSE_CHECKARG_ENUM(3, trans_B);

        ROCSPARSE_CHECKARG(9,
                           descr,
                           (descr->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);

        ROCSPARSE_CHECKARG(9,
                           descr,
                           (descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);

        ROCSPARSE_CHECKARG(
            2, trans_A, (trans_A != rocsparse_operation_none), rocsparse_status_not_implemented);

        ROCSPARSE_CHECKARG(
            3,
            trans_B,
            (trans_B != rocsparse_operation_none && trans_B != rocsparse_operation_transpose),
            rocsparse_status_not_implemented);

        ROCSPARSE_CHECKARG_SIZE(4, mb);
        ROCSPARSE_CHECKARG_SIZE(5, n);
        ROCSPARSE_CHECKARG_SIZE(6, kb);
        ROCSPARSE_CHECKARG_SIZE(7, nnzb);
        ROCSPARSE_CHECKARG_SIZE(13, row_block_dim);
        ROCSPARSE_CHECKARG_SIZE(14, col_block_dim);
        ROCSPARSE_CHECKARG(13, row_block_dim, (row_block_dim == 0), rocsparse_status_invalid_size);
        ROCSPARSE_CHECKARG(14, col_block_dim, (col_block_dim == 0), rocsparse_status_invalid_size);
        ROCSPARSE_CHECKARG_ARRAY(10, nnzb, bsr_val);
        ROCSPARSE_CHECKARG_ARRAY(11, mb, bsr_row_ptr);
        ROCSPARSE_CHECKARG_ARRAY(12, nnzb, bsr_col_ind);

        //
        // quick return.
        //
        const rocsparse_status status = rocsparse::gebsrmm_quickreturn(handle,
                                                                       dir,
                                                                       trans_A,
                                                                       trans_B,
                                                                       mb,
                                                                       n,
                                                                       kb,
                                                                       nnzb,
                                                                       alpha,
                                                                       descr,
                                                                       bsr_val,
                                                                       bsr_row_ptr,
                                                                       bsr_col_ind,
                                                                       row_block_dim,
                                                                       col_block_dim,
                                                                       B,
                                                                       ldb,
                                                                       beta,
                                                                       C,
                                                                       ldc);

        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG_POINTER(8, alpha);
        ROCSPARSE_CHECKARG_POINTER(15, B);
        ROCSPARSE_CHECKARG_SIZE(16, ldb);
        ROCSPARSE_CHECKARG_POINTER(17, beta);
        ROCSPARSE_CHECKARG_POINTER(18, C);
        ROCSPARSE_CHECKARG_SIZE(19, ldc);

        static constexpr rocsparse_int s_one = static_cast<rocsparse_int>(1);
        switch(trans_A)
        {
        case rocsparse_operation_none:
        {
            ROCSPARSE_CHECKARG(19,
                               ldc,
                               (ldc < rocsparse::max(s_one, mb * row_block_dim)),
                               rocsparse_status_invalid_size);

            // Check leading dimension of B
            switch(trans_B)
            {
            case rocsparse_operation_none:
            {
                ROCSPARSE_CHECKARG(16,
                                   ldb,
                                   (ldb < rocsparse::max(s_one, kb * col_block_dim)),
                                   rocsparse_status_invalid_size);
                break;
            }
            case rocsparse_operation_transpose:
            case rocsparse_operation_conjugate_transpose:
            {
                ROCSPARSE_CHECKARG(
                    16, ldb, (ldb < rocsparse::max(s_one, n)), rocsparse_status_invalid_size);
                break;
            }
            }
            break;
        }
        case rocsparse_operation_transpose:
        case rocsparse_operation_conjugate_transpose:
        {
            ROCSPARSE_CHECKARG(19,
                               ldc,
                               (ldc < rocsparse::max(s_one, kb * col_block_dim)),
                               rocsparse_status_invalid_size);

            switch(trans_B)
            {
            case rocsparse_operation_none:
            {
                ROCSPARSE_CHECKARG(16,
                                   ldb,
                                   (ldb < rocsparse::max(s_one, mb * row_block_dim)),
                                   rocsparse_status_invalid_size);
                break;
            }
            case rocsparse_operation_transpose:
            case rocsparse_operation_conjugate_transpose:
            {
                ROCSPARSE_CHECKARG(
                    16, ldb, (ldb < rocsparse::max(s_one, n)), rocsparse_status_invalid_size);
                break;
            }
            }
            break;
        }
        }

        return rocsparse_status_continue;
    }

    template <typename T>
    static rocsparse_status gebsrmm_core(rocsparse_handle          handle,
                                         rocsparse_direction       dir,
                                         rocsparse_operation       trans_A,
                                         rocsparse_operation       trans_B,
                                         rocsparse_int             mb,
                                         rocsparse_int             n,
                                         rocsparse_int             kb,
                                         rocsparse_int             nnzb,
                                         const T*                  alpha,
                                         const rocsparse_mat_descr descr,
                                         const T*                  bsr_val,
                                         const rocsparse_int*      bsr_row_ptr,
                                         const rocsparse_int*      bsr_col_ind,
                                         rocsparse_int             row_block_dim,
                                         rocsparse_int             col_block_dim,
                                         const T*                  B,
                                         rocsparse_int             ldb,
                                         const T*                  beta,
                                         T*                        C,
                                         rocsparse_int             ldc)
    {
        ROCSPARSE_ROUTINE_TRACE;

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gebsrmm_template_dispatch(handle,
                                                                       dir,
                                                                       trans_A,
                                                                       trans_B,
                                                                       mb,
                                                                       n,
                                                                       kb,
                                                                       nnzb,
                                                                       alpha,
                                                                       descr,
                                                                       bsr_val,
                                                                       bsr_row_ptr,
                                                                       bsr_col_ind,
                                                                       row_block_dim,
                                                                       col_block_dim,
                                                                       B,
                                                                       ldb,
                                                                       beta,
                                                                       C,
                                                                       ldc));
        return rocsparse_status_success;
    }

    template <typename... P>
    rocsparse_status gebsrmm_template(P&&... p)
    {
        const rocsparse_status status = rocsparse::gebsrmm_quickreturn(p...);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gebsrmm_core(p...));
        return rocsparse_status_success;
    }

    template <typename T>
    rocsparse_status gebsrmm_impl(rocsparse_handle          handle,
                                  rocsparse_direction       dir,
                                  rocsparse_operation       trans_A,
                                  rocsparse_operation       trans_B,
                                  rocsparse_int             mb,
                                  rocsparse_int             n,
                                  rocsparse_int             kb,
                                  rocsparse_int             nnzb,
                                  const T*                  alpha,
                                  const rocsparse_mat_descr descr,
                                  const T*                  bsr_val,
                                  const rocsparse_int*      bsr_row_ptr,
                                  const rocsparse_int*      bsr_col_ind,
                                  rocsparse_int             row_block_dim,
                                  rocsparse_int             col_block_dim,
                                  const T*                  B,
                                  rocsparse_int             ldb,
                                  const T*                  beta,
                                  T*                        C,
                                  rocsparse_int             ldc)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Logging TODO bench logging
        rocsparse::log_trace(handle,
                             rocsparse::replaceX<T>("rocsparse_Xgebsrmm"),
                             dir,
                             trans_A,
                             trans_B,
                             mb,
                             n,
                             kb,
                             nnzb,
                             LOG_TRACE_SCALAR_VALUE(handle, alpha),
                             (const void*&)descr,
                             (const void*&)bsr_val,
                             (const void*&)bsr_row_ptr,
                             (const void*&)bsr_col_ind,
                             row_block_dim,
                             col_block_dim,
                             (const void*&)B,
                             ldb,
                             LOG_TRACE_SCALAR_VALUE(handle, beta),
                             (const void*&)C,
                             ldc);

        const rocsparse_status status = rocsparse::gebsrmm_checkarg(handle,
                                                                    dir,
                                                                    trans_A,
                                                                    trans_B,
                                                                    mb,
                                                                    n,
                                                                    kb,
                                                                    nnzb,
                                                                    alpha,
                                                                    descr,
                                                                    bsr_val,
                                                                    bsr_row_ptr,
                                                                    bsr_col_ind,
                                                                    row_block_dim,
                                                                    col_block_dim,
                                                                    B,
                                                                    ldb,
                                                                    beta,
                                                                    C,
                                                                    ldc);

        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gebsrmm_core(handle,
                                                          dir,
                                                          trans_A,
                                                          trans_B,
                                                          mb,
                                                          n,
                                                          kb,
                                                          nnzb,
                                                          alpha,
                                                          descr,
                                                          bsr_val,
                                                          bsr_row_ptr,
                                                          bsr_col_ind,
                                                          row_block_dim,
                                                          col_block_dim,
                                                          B,
                                                          ldb,
                                                          beta,
                                                          C,
                                                          ldc));

        return rocsparse_status_success;
    }
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
#define C_IMPL(NAME, TYPE)                                                    \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,        \
                                     rocsparse_direction       dir,           \
                                     rocsparse_operation       trans_A,       \
                                     rocsparse_operation       trans_B,       \
                                     rocsparse_int             mb,            \
                                     rocsparse_int             n,             \
                                     rocsparse_int             kb,            \
                                     rocsparse_int             nnzb,          \
                                     const TYPE*               alpha,         \
                                     const rocsparse_mat_descr descr,         \
                                     const TYPE*               bsr_val,       \
                                     const rocsparse_int*      bsr_row_ptr,   \
                                     const rocsparse_int*      bsr_col_ind,   \
                                     rocsparse_int             row_block_dim, \
                                     rocsparse_int             col_block_dim, \
                                     const TYPE*               B,             \
                                     rocsparse_int             ldb,           \
                                     const TYPE*               beta,          \
                                     TYPE*                     C,             \
                                     rocsparse_int             ldc)           \
    try                                                                       \
    {                                                                         \
        ROCSPARSE_ROUTINE_TRACE;                                              \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gebsrmm_impl(handle,             \
                                                          dir,                \
                                                          trans_A,            \
                                                          trans_B,            \
                                                          mb,                 \
                                                          n,                  \
                                                          kb,                 \
                                                          nnzb,               \
                                                          alpha,              \
                                                          descr,              \
                                                          bsr_val,            \
                                                          bsr_row_ptr,        \
                                                          bsr_col_ind,        \
                                                          row_block_dim,      \
                                                          col_block_dim,      \
                                                          B,                  \
                                                          ldb,                \
                                                          beta,               \
                                                          C,                  \
                                                          ldc));              \
        return rocsparse_status_success;                                      \
    }                                                                         \
    catch(...)                                                                \
    {                                                                         \
        RETURN_ROCSPARSE_EXCEPTION();                                         \
    }

C_IMPL(rocsparse_sgebsrmm, float);
C_IMPL(rocsparse_dgebsrmm, double);
C_IMPL(rocsparse_cgebsrmm, rocsparse_float_complex);
C_IMPL(rocsparse_zgebsrmm, rocsparse_double_complex);

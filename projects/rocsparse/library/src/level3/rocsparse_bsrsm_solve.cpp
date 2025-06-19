/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "../level2/rocsparse_bsrsv.hpp"
#include "internal/level3/rocsparse_bsrsm.h"
#include "rocsparse_bsrsm.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <typename T>
    rocsparse_status bsrsm_solve_template_large(rocsparse_handle          handle,
                                                rocsparse_direction       dir,
                                                rocsparse_operation       trans_A,
                                                rocsparse_operation       trans_X,
                                                rocsparse_int             mb,
                                                rocsparse_int             nrhs,
                                                rocsparse_int             nnzb,
                                                const T*                  alpha,
                                                const rocsparse_mat_descr descr,
                                                const T*                  bsr_val,
                                                const rocsparse_int*      bsr_row_ptr,
                                                const rocsparse_int*      bsr_col_ind,
                                                rocsparse_int             block_dim,
                                                rocsparse_mat_info        info,
                                                const T*                  B,
                                                int64_t                   ldb,
                                                T*                        X,
                                                int64_t                   ldx,
                                                void*                     temp_buffer);

    template <typename T>
    static rocsparse_status bsrsm_solve_template_dispatch(rocsparse_handle          handle,
                                                          rocsparse_direction       dir,
                                                          rocsparse_operation       trans_A,
                                                          rocsparse_operation       trans_X,
                                                          rocsparse_int             mb,
                                                          rocsparse_int             nrhs,
                                                          rocsparse_int             nnzb,
                                                          const T*                  alpha,
                                                          const rocsparse_mat_descr descr,
                                                          const T*                  bsr_val,
                                                          const rocsparse_int*      bsr_row_ptr,
                                                          const rocsparse_int*      bsr_col_ind,
                                                          rocsparse_int             block_dim,
                                                          rocsparse_mat_info        info,
                                                          const T*                  B,
                                                          int64_t                   ldb,
                                                          T*                        X,
                                                          int64_t                   ldx,
                                                          rocsparse_solve_policy    policy,
                                                          void*                     temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrsm_solve_template_large(handle,
                                                                        dir,
                                                                        trans_A,
                                                                        trans_X,
                                                                        mb,
                                                                        nrhs,
                                                                        nnzb,
                                                                        alpha,
                                                                        descr,
                                                                        bsr_val,
                                                                        bsr_row_ptr,
                                                                        bsr_col_ind,
                                                                        block_dim,
                                                                        info,
                                                                        B,
                                                                        ldb,
                                                                        X,
                                                                        ldx,
                                                                        temp_buffer));
        return rocsparse_status_success;
    }
}

rocsparse_status rocsparse::bsrsm_solve_quickreturn(rocsparse_handle          handle,
                                                    rocsparse_direction       dir,
                                                    rocsparse_operation       trans_A,
                                                    rocsparse_operation       trans_X,
                                                    rocsparse_int             mb,
                                                    rocsparse_int             nrhs,
                                                    rocsparse_int             nnzb,
                                                    const void*               alpha,
                                                    const rocsparse_mat_descr descr,
                                                    const void*               bsr_val,
                                                    const rocsparse_int*      bsr_row_ptr,
                                                    const rocsparse_int*      bsr_col_ind,
                                                    rocsparse_int             block_dim,
                                                    rocsparse_mat_info        info,
                                                    const void*               B,
                                                    int64_t                   ldb,
                                                    void*                     X,
                                                    int64_t                   ldx,
                                                    rocsparse_solve_policy    policy,
                                                    void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(mb == 0 || nrhs == 0)
    {
        return rocsparse_status_success;
    }
    return rocsparse_status_continue;
}

namespace rocsparse
{
    static rocsparse_status bsrsm_solve_checkarg(rocsparse_handle          handle, //0
                                                 rocsparse_direction       dir, //1
                                                 rocsparse_operation       trans_A, //2
                                                 rocsparse_operation       trans_X, //3
                                                 rocsparse_int             mb, //4
                                                 rocsparse_int             nrhs, //5
                                                 rocsparse_int             nnzb, //6
                                                 const void*               alpha, //7
                                                 const rocsparse_mat_descr descr, //8
                                                 const void*               bsr_val, //9
                                                 const rocsparse_int*      bsr_row_ptr, //10
                                                 const rocsparse_int*      bsr_col_ind, //11
                                                 rocsparse_int             block_dim, //12
                                                 rocsparse_mat_info        info, //13
                                                 const void*               B, //14
                                                 int64_t                   ldb, //15
                                                 void*                     X, //16
                                                 int64_t                   ldx, //17
                                                 rocsparse_solve_policy    policy, //18
                                                 void*                     temp_buffer) //19
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_ENUM(1, dir);
        ROCSPARSE_CHECKARG_ENUM(2, trans_A);
        ROCSPARSE_CHECKARG_ENUM(3, trans_X);
        ROCSPARSE_CHECKARG_SIZE(4, mb);
        ROCSPARSE_CHECKARG_SIZE(5, nrhs);

        const rocsparse_status status = rocsparse::bsrsm_solve_quickreturn(handle,
                                                                           dir,
                                                                           trans_A,
                                                                           trans_X,
                                                                           mb,
                                                                           nrhs,
                                                                           nnzb,
                                                                           alpha,
                                                                           descr,
                                                                           bsr_val,
                                                                           bsr_row_ptr,
                                                                           bsr_col_ind,
                                                                           block_dim,
                                                                           info,
                                                                           B,
                                                                           ldb,
                                                                           X,
                                                                           ldx,
                                                                           policy,
                                                                           temp_buffer);

        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG_SIZE(6, nnzb);
        ROCSPARSE_CHECKARG_POINTER(7, alpha);
        ROCSPARSE_CHECKARG_POINTER(8, descr);
        ROCSPARSE_CHECKARG(8,
                           descr,
                           (descr->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(8,
                           descr,
                           (descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);
        ROCSPARSE_CHECKARG_ARRAY(9, nnzb, bsr_val);
        ROCSPARSE_CHECKARG_ARRAY(10, mb, bsr_row_ptr);
        ROCSPARSE_CHECKARG_ARRAY(11, nnzb, bsr_col_ind);
        ROCSPARSE_CHECKARG_SIZE(12, block_dim);
        ROCSPARSE_CHECKARG(12, block_dim, (block_dim == 0), rocsparse_status_invalid_size);
        ROCSPARSE_CHECKARG_POINTER(13, info);
        ROCSPARSE_CHECKARG_POINTER(14, B);
        ROCSPARSE_CHECKARG_SIZE(15, ldb);
        ROCSPARSE_CHECKARG(15,
                           ldb,
                           ((trans_X == rocsparse_operation_none) && (ldb < mb * block_dim)),
                           rocsparse_status_invalid_size);
        ROCSPARSE_CHECKARG(15,
                           ldb,
                           ((trans_X != rocsparse_operation_none) && (ldb < nrhs)),
                           rocsparse_status_invalid_size);
        ROCSPARSE_CHECKARG_POINTER(16, X);
        ROCSPARSE_CHECKARG_SIZE(17, ldx);
        ROCSPARSE_CHECKARG(17,
                           ldx,
                           ((trans_X == rocsparse_operation_none) && (ldx < mb * block_dim)),
                           rocsparse_status_invalid_size);
        ROCSPARSE_CHECKARG(17,
                           ldx,
                           ((trans_X != rocsparse_operation_none) && (ldx < nrhs)),
                           rocsparse_status_invalid_size);
        ROCSPARSE_CHECKARG_ENUM(18, policy);

        ROCSPARSE_CHECKARG_POINTER(19, temp_buffer);

        return rocsparse_status_continue;
    }
}

template <typename T>
rocsparse_status rocsparse::bsrsm_solve_core(rocsparse_handle          handle,
                                             rocsparse_direction       dir,
                                             rocsparse_operation       trans_A,
                                             rocsparse_operation       trans_X,
                                             rocsparse_int             mb,
                                             rocsparse_int             nrhs,
                                             rocsparse_int             nnzb,
                                             const T*                  alpha,
                                             const rocsparse_mat_descr descr,
                                             const T*                  bsr_val,
                                             const rocsparse_int*      bsr_row_ptr,
                                             const rocsparse_int*      bsr_col_ind,
                                             rocsparse_int             block_dim,
                                             rocsparse_mat_info        info,
                                             const T*                  B,
                                             int64_t                   ldb,
                                             T*                        X,
                                             int64_t                   ldx,
                                             rocsparse_solve_policy    policy,
                                             void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrsm_solve_template_dispatch(handle,
                                                                       dir,
                                                                       trans_A,
                                                                       trans_X,
                                                                       mb,
                                                                       nrhs,
                                                                       nnzb,
                                                                       alpha,
                                                                       descr,
                                                                       bsr_val,
                                                                       bsr_row_ptr,
                                                                       bsr_col_ind,
                                                                       block_dim,
                                                                       info,
                                                                       B,
                                                                       ldb,
                                                                       X,
                                                                       ldx,
                                                                       policy,
                                                                       temp_buffer));
    return rocsparse_status_success;
}

namespace rocsparse
{
    template <typename T>
    rocsparse_status bsrsm_solve_impl(rocsparse_handle          handle,
                                      rocsparse_direction       dir,
                                      rocsparse_operation       trans_A,
                                      rocsparse_operation       trans_X,
                                      rocsparse_int             mb,
                                      rocsparse_int             nrhs,
                                      rocsparse_int             nnzb,
                                      const T*                  alpha,
                                      const rocsparse_mat_descr descr,
                                      const T*                  bsr_val,
                                      const rocsparse_int*      bsr_row_ptr,
                                      const rocsparse_int*      bsr_col_ind,
                                      rocsparse_int             block_dim,
                                      rocsparse_mat_info        info,
                                      const T*                  B,
                                      int64_t                   ldb,
                                      T*                        X,
                                      int64_t                   ldx,
                                      rocsparse_solve_policy    policy,
                                      void*                     temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        rocsparse::log_trace(handle,
                             rocsparse::replaceX<T>("rocsparse_Xbsrsm_solve"),
                             dir,
                             trans_A,
                             trans_X,
                             mb,
                             nrhs,
                             nnzb,
                             LOG_TRACE_SCALAR_VALUE(handle, alpha),
                             (const void*&)descr,
                             (const void*&)bsr_val,
                             (const void*&)bsr_row_ptr,
                             (const void*&)bsr_col_ind,
                             block_dim,
                             (const void*&)info,
                             (const void*&)B,
                             ldb,
                             (const void*&)X,
                             ldx,
                             policy,
                             (const void*&)temp_buffer);

        const rocsparse_status status = rocsparse::bsrsm_solve_checkarg(handle,
                                                                        dir,
                                                                        trans_A,
                                                                        trans_X,
                                                                        mb,
                                                                        nrhs,
                                                                        nnzb,
                                                                        alpha,
                                                                        descr,
                                                                        bsr_val,
                                                                        bsr_row_ptr,
                                                                        bsr_col_ind,
                                                                        block_dim,
                                                                        info,
                                                                        B,
                                                                        ldb,
                                                                        X,
                                                                        ldx,
                                                                        policy,
                                                                        temp_buffer);

        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrsm_solve_core(handle,
                                                              dir,
                                                              trans_A,
                                                              trans_X,
                                                              mb,
                                                              nrhs,
                                                              nnzb,
                                                              alpha,
                                                              descr,
                                                              bsr_val,
                                                              bsr_row_ptr,
                                                              bsr_col_ind,
                                                              block_dim,
                                                              info,
                                                              B,
                                                              ldb,
                                                              X,
                                                              ldx,
                                                              policy,
                                                              temp_buffer));
        return rocsparse_status_success;
    }
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
#define C_IMPL(NAME, TYPE)                                                   \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,       \
                                     rocsparse_direction       dir,          \
                                     rocsparse_operation       trans_A,      \
                                     rocsparse_operation       trans_X,      \
                                     rocsparse_int             mb,           \
                                     rocsparse_int             nrhs,         \
                                     rocsparse_int             nnzb,         \
                                     const TYPE*               alpha,        \
                                     const rocsparse_mat_descr descr,        \
                                     const TYPE*               bsr_val,      \
                                     const rocsparse_int*      bsr_row_ptr,  \
                                     const rocsparse_int*      bsr_col_ind,  \
                                     rocsparse_int             block_dim,    \
                                     rocsparse_mat_info        info,         \
                                     const TYPE*               B,            \
                                     rocsparse_int             ldb,          \
                                     TYPE*                     X,            \
                                     rocsparse_int             ldx,          \
                                     rocsparse_solve_policy    policy,       \
                                     void*                     temp_buffer)  \
    try                                                                      \
    {                                                                        \
        ROCSPARSE_ROUTINE_TRACE;                                             \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrsm_solve_impl(handle,        \
                                                              dir,           \
                                                              trans_A,       \
                                                              trans_X,       \
                                                              mb,            \
                                                              nrhs,          \
                                                              nnzb,          \
                                                              alpha,         \
                                                              descr,         \
                                                              bsr_val,       \
                                                              bsr_row_ptr,   \
                                                              bsr_col_ind,   \
                                                              block_dim,     \
                                                              info,          \
                                                              B,             \
                                                              ldb,           \
                                                              X,             \
                                                              ldx,           \
                                                              policy,        \
                                                              temp_buffer)); \
        return rocsparse_status_success;                                     \
    }                                                                        \
    catch(...)                                                               \
    {                                                                        \
        RETURN_ROCSPARSE_EXCEPTION();                                        \
    }

C_IMPL(rocsparse_sbsrsm_solve, float);
C_IMPL(rocsparse_dbsrsm_solve, double);
C_IMPL(rocsparse_cbsrsm_solve, rocsparse_float_complex);
C_IMPL(rocsparse_zbsrsm_solve, rocsparse_double_complex);

#undef C_IMPL

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

#include "internal/level2/rocsparse_gebsrmv.h"
#include "rocsparse_bsrmv.hpp"
#include "rocsparse_csrmv.hpp"
#include "rocsparse_gebsrmv.hpp"

#include "gebsrmv_device.h"
#include "rocsparse_common.h"
#include "rocsparse_control.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_utility.hpp"

#include <hip/hip_runtime.h>

namespace rocsparse
{
    template <typename T>
    rocsparse_status gebsrmv_template_row_block_dim_1(rocsparse_handle          handle,
                                                      rocsparse_direction       dir,
                                                      rocsparse_operation       trans,
                                                      rocsparse_int             mb,
                                                      rocsparse_int             nb,
                                                      rocsparse_int             nnzb,
                                                      const T*                  alpha,
                                                      const rocsparse_mat_descr descr,
                                                      const T*                  bsr_val,
                                                      const rocsparse_int*      bsr_row_ptr,
                                                      const rocsparse_int*      bsr_col_ind,
                                                      rocsparse_int             row_block_dim,
                                                      rocsparse_int             col_block_dim,
                                                      const T*                  x,
                                                      const T*                  beta,
                                                      T*                        y);

    template <typename T>
    rocsparse_status gebsrmv_template_row_block_dim_2(rocsparse_handle          handle,
                                                      rocsparse_direction       dir,
                                                      rocsparse_operation       trans,
                                                      rocsparse_int             mb,
                                                      rocsparse_int             nb,
                                                      rocsparse_int             nnzb,
                                                      const T*                  alpha,
                                                      const rocsparse_mat_descr descr,
                                                      const T*                  bsr_val,
                                                      const rocsparse_int*      bsr_row_ptr,
                                                      const rocsparse_int*      bsr_col_ind,
                                                      rocsparse_int             row_block_dim,
                                                      rocsparse_int             col_block_dim,
                                                      const T*                  x,
                                                      const T*                  beta,
                                                      T*                        y);

    template <typename T>
    rocsparse_status gebsrmv_template_row_block_dim_3(rocsparse_handle          handle,
                                                      rocsparse_direction       dir,
                                                      rocsparse_operation       trans,
                                                      rocsparse_int             mb,
                                                      rocsparse_int             nb,
                                                      rocsparse_int             nnzb,
                                                      const T*                  alpha,
                                                      const rocsparse_mat_descr descr,
                                                      const T*                  bsr_val,
                                                      const rocsparse_int*      bsr_row_ptr,
                                                      const rocsparse_int*      bsr_col_ind,
                                                      rocsparse_int             row_block_dim,
                                                      rocsparse_int             col_block_dim,
                                                      const T*                  x,
                                                      const T*                  beta,
                                                      T*                        y);

    template <typename T>
    rocsparse_status gebsrmv_template_row_block_dim_4(rocsparse_handle          handle,
                                                      rocsparse_direction       dir,
                                                      rocsparse_operation       trans,
                                                      rocsparse_int             mb,
                                                      rocsparse_int             nb,
                                                      rocsparse_int             nnzb,
                                                      const T*                  alpha,
                                                      const rocsparse_mat_descr descr,
                                                      const T*                  bsr_val,
                                                      const rocsparse_int*      bsr_row_ptr,
                                                      const rocsparse_int*      bsr_col_ind,
                                                      rocsparse_int             row_block_dim,
                                                      rocsparse_int             col_block_dim,
                                                      const T*                  x,
                                                      const T*                  beta,
                                                      T*                        y);

    template <typename T>
    rocsparse_status gebsrmv_template_row_block_dim_5_8(rocsparse_handle          handle,
                                                        rocsparse_direction       dir,
                                                        rocsparse_operation       trans,
                                                        rocsparse_int             mb,
                                                        rocsparse_int             nb,
                                                        rocsparse_int             nnzb,
                                                        const T*                  alpha,
                                                        const rocsparse_mat_descr descr,
                                                        const T*                  bsr_val,
                                                        const rocsparse_int*      bsr_row_ptr,
                                                        const rocsparse_int*      bsr_col_ind,
                                                        rocsparse_int             row_block_dim,
                                                        rocsparse_int             col_block_dim,
                                                        const T*                  x,
                                                        const T*                  beta,
                                                        T*                        y);

    template <typename T>
    rocsparse_status gebsrmv_template_row_block_dim_9_12(rocsparse_handle          handle,
                                                         rocsparse_direction       dir,
                                                         rocsparse_operation       trans,
                                                         rocsparse_int             mb,
                                                         rocsparse_int             nb,
                                                         rocsparse_int             nnzb,
                                                         const T*                  alpha,
                                                         const rocsparse_mat_descr descr,
                                                         const T*                  bsr_val,
                                                         const rocsparse_int*      bsr_row_ptr,
                                                         const rocsparse_int*      bsr_col_ind,
                                                         rocsparse_int             row_block_dim,
                                                         rocsparse_int             col_block_dim,
                                                         const T*                  x,
                                                         const T*                  beta,
                                                         T*                        y);

    template <typename T>
    rocsparse_status gebsrmv_template_row_block_dim_13_16(rocsparse_handle          handle,
                                                          rocsparse_direction       dir,
                                                          rocsparse_operation       trans,
                                                          rocsparse_int             mb,
                                                          rocsparse_int             nb,
                                                          rocsparse_int             nnzb,
                                                          const T*                  alpha,
                                                          const rocsparse_mat_descr descr,
                                                          const T*                  bsr_val,
                                                          const rocsparse_int*      bsr_row_ptr,
                                                          const rocsparse_int*      bsr_col_ind,
                                                          rocsparse_int             row_block_dim,
                                                          rocsparse_int             col_block_dim,
                                                          const T*                  x,
                                                          const T*                  beta,
                                                          T*                        y);

    template <typename T>
    rocsparse_status gebsrmv_template_row_block_dim_17_inf(rocsparse_handle          handle,
                                                           rocsparse_direction       dir,
                                                           rocsparse_operation       trans,
                                                           rocsparse_int             mb,
                                                           rocsparse_int             nb,
                                                           rocsparse_int             nnzb,
                                                           const T*                  alpha,
                                                           const rocsparse_mat_descr descr,
                                                           const T*                  bsr_val,
                                                           const rocsparse_int*      bsr_row_ptr,
                                                           const rocsparse_int*      bsr_col_ind,
                                                           rocsparse_int             row_block_dim,
                                                           rocsparse_int             col_block_dim,
                                                           const T*                  x,
                                                           const T*                  beta,
                                                           T*                        y);

    template <typename... Ts>
    rocsparse_status gebsrmv_template_dispatch_specialization(rocsparse_int row_block_dim,
                                                              Ts&&... ts)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if(row_block_dim == 1)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::gebsrmv_template_row_block_dim_1(ts...));
            return rocsparse_status_success;
        }
        else if(row_block_dim == 2)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::gebsrmv_template_row_block_dim_2(ts...));
            return rocsparse_status_success;
        }
        else if(row_block_dim == 3)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::gebsrmv_template_row_block_dim_3(ts...));
            return rocsparse_status_success;
        }
        else if(row_block_dim == 4)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::gebsrmv_template_row_block_dim_4(ts...));
            return rocsparse_status_success;
        }
        else if(row_block_dim <= 8)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::gebsrmv_template_row_block_dim_5_8(ts...));
            return rocsparse_status_success;
        }
        else if(row_block_dim <= 12)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::gebsrmv_template_row_block_dim_9_12(ts...));
            return rocsparse_status_success;
        }
        else if(row_block_dim <= 16)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::gebsrmv_template_row_block_dim_13_16(ts...));
            return rocsparse_status_success;
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::gebsrmv_template_row_block_dim_17_inf(ts...));
            return rocsparse_status_success;
        }
    }
}

template <typename T>
rocsparse_status rocsparse::gebsrmv_template_dispatch(rocsparse_handle          handle,
                                                      rocsparse_direction       dir,
                                                      rocsparse_operation       trans,
                                                      rocsparse_int             mb,
                                                      rocsparse_int             nb,
                                                      rocsparse_int             nnzb,
                                                      const T*                  alpha,
                                                      const rocsparse_mat_descr descr,
                                                      const T*                  bsr_val,
                                                      const rocsparse_int*      bsr_row_ptr,
                                                      const rocsparse_int*      bsr_col_ind,
                                                      rocsparse_int             row_block_dim,
                                                      rocsparse_int             col_block_dim,
                                                      const T*                  x,
                                                      const T*                  beta,
                                                      T*                        y)
{
    ROCSPARSE_ROUTINE_TRACE;

    // row_block_dim == col_block_dim is the BSR case
    if(row_block_dim == col_block_dim)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrmv_template_dispatch(handle,
                                                                     dir,
                                                                     trans,
                                                                     mb,
                                                                     nb,
                                                                     nnzb,
                                                                     alpha,
                                                                     descr,
                                                                     bsr_val,
                                                                     bsr_row_ptr,
                                                                     bsr_col_ind,
                                                                     row_block_dim,
                                                                     x,
                                                                     beta,
                                                                     y));

        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gebsrmv_template_dispatch_specialization(row_block_dim,
                                                                                  //
                                                                                  handle,
                                                                                  dir,
                                                                                  trans,
                                                                                  mb,
                                                                                  nb,
                                                                                  nnzb,
                                                                                  alpha,
                                                                                  descr,
                                                                                  bsr_val,
                                                                                  bsr_row_ptr,
                                                                                  bsr_col_ind,
                                                                                  row_block_dim,
                                                                                  col_block_dim,
                                                                                  x,
                                                                                  beta,
                                                                                  y));
    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse::gebsrmv_template(rocsparse_handle          handle, //0
                                             rocsparse_direction       dir, //1
                                             rocsparse_operation       trans, //2
                                             rocsparse_int             mb, //3
                                             rocsparse_int             nb, //4
                                             rocsparse_int             nnzb, //5
                                             const T*                  alpha, //6
                                             const rocsparse_mat_descr descr, //7
                                             const T*                  bsr_val, //8
                                             const rocsparse_int*      bsr_row_ptr, //9
                                             const rocsparse_int*      bsr_col_ind, //10
                                             rocsparse_int             row_block_dim, //11
                                             rocsparse_int             col_block_dim, //12
                                             const T*                  x, //13
                                             const T*                  beta, //14
                                             T*                        y) //15
{
    ROCSPARSE_ROUTINE_TRACE;

    // Check for valid handle and matrix descriptor
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(7, descr);

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xgebsrmv"),
                         dir,
                         trans,
                         mb,
                         nb,
                         nnzb,
                         LOG_TRACE_SCALAR_VALUE(handle, alpha),
                         (const void*&)descr,
                         (const void*&)bsr_val,
                         (const void*&)bsr_row_ptr,
                         (const void*&)bsr_col_ind,
                         row_block_dim,
                         col_block_dim,
                         (const void*&)x,
                         LOG_TRACE_SCALAR_VALUE(handle, beta),
                         (const void*&)y);

    ROCSPARSE_CHECKARG_ENUM(1, dir);
    ROCSPARSE_CHECKARG_ENUM(2, trans);
    ROCSPARSE_CHECKARG(
        2, trans, (trans != rocsparse_operation_none), rocsparse_status_not_implemented);

    // Check matrix type
    ROCSPARSE_CHECKARG(
        7, descr, (descr->type != rocsparse_matrix_type_general), rocsparse_status_not_implemented);

    // Check matrix sorting mode

    ROCSPARSE_CHECKARG(7,
                       descr,
                       (descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    // Check sizes
    ROCSPARSE_CHECKARG_SIZE(3, mb);
    ROCSPARSE_CHECKARG_SIZE(4, nb);
    ROCSPARSE_CHECKARG_SIZE(5, nnzb);

    ROCSPARSE_CHECKARG_SIZE(11, row_block_dim);
    ROCSPARSE_CHECKARG(11, row_block_dim, (row_block_dim == 0), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_SIZE(12, col_block_dim);
    ROCSPARSE_CHECKARG(12, col_block_dim, (col_block_dim == 0), rocsparse_status_invalid_size);
    // Quick return if possible
    if(mb == 0 || nb == 0)
    {
        // matrix never accessed however still need to update y vector
        rocsparse_int ysize
            = (trans == rocsparse_operation_none) ? row_block_dim * mb : col_block_dim * nb;
        if(ysize > 0)
        {
            if(y == nullptr && beta == nullptr)
            {
                return rocsparse_status_invalid_pointer;
            }

            RETURN_IF_ROCSPARSE_ERROR(rocsparse::scale_array(handle, ysize, beta, y));
        }

        return rocsparse_status_success;
    }

    // Check pointer arguments
    ROCSPARSE_CHECKARG_POINTER(6, alpha);

    const rocsparse_int xsize = (trans == rocsparse_operation_none) ? nb : mb;
    const rocsparse_int ysize = (trans == rocsparse_operation_none) ? mb : nb;

    ROCSPARSE_CHECKARG_ARRAY(8, nnzb, bsr_val);
    ROCSPARSE_CHECKARG_ARRAY(9, mb, bsr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(10, nnzb, bsr_col_ind);
    ROCSPARSE_CHECKARG_ARRAY(13, xsize, x);

    ROCSPARSE_CHECKARG_POINTER(14, beta);

    ROCSPARSE_CHECKARG_ARRAY(15, ysize, y);

    // row_block_dim == 1 and col_block_dim == 1 is the CSR case
    if(row_block_dim == 1 && col_block_dim == 1)
    {
        static constexpr bool fallback_algorithm = true;
        static constexpr bool force_conj         = false;

        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::csrmv_template<T, rocsparse_int, rocsparse_int, T, T, T>(
                handle,
                trans,
                rocsparse::csrmv_alg_rowsplit,
                mb,
                nb,
                nnzb,
                alpha,
                descr,
                bsr_val,
                bsr_row_ptr,
                bsr_row_ptr + 1,
                bsr_col_ind,
                nullptr,
                x,
                beta,
                y,
                force_conj,
                fallback_algorithm)));
        return rocsparse_status_success;
    }

    // Run different gebsrmv kernels
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gebsrmv_template_dispatch(handle,
                                                                   dir,
                                                                   trans,
                                                                   mb,
                                                                   nb,
                                                                   nnzb,
                                                                   alpha,
                                                                   descr,
                                                                   bsr_val,
                                                                   bsr_row_ptr,
                                                                   bsr_col_ind,
                                                                   row_block_dim,
                                                                   col_block_dim,
                                                                   x,
                                                                   beta,
                                                                   y));
    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
#define C_IMPL(NAME, TYPE)                                                    \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,        \
                                     rocsparse_direction       dir,           \
                                     rocsparse_operation       trans,         \
                                     rocsparse_int             mb,            \
                                     rocsparse_int             nb,            \
                                     rocsparse_int             nnzb,          \
                                     const TYPE*               alpha,         \
                                     const rocsparse_mat_descr descr,         \
                                     const TYPE*               bsr_val,       \
                                     const rocsparse_int*      bsr_row_ptr,   \
                                     const rocsparse_int*      bsr_col_ind,   \
                                     rocsparse_int             row_block_dim, \
                                     rocsparse_int             col_block_dim, \
                                     const TYPE*               x,             \
                                     const TYPE*               beta,          \
                                     TYPE*                     y)             \
    try                                                                       \
    {                                                                         \
        ROCSPARSE_ROUTINE_TRACE;                                              \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gebsrmv_template(handle,         \
                                                              dir,            \
                                                              trans,          \
                                                              mb,             \
                                                              nb,             \
                                                              nnzb,           \
                                                              alpha,          \
                                                              descr,          \
                                                              bsr_val,        \
                                                              bsr_row_ptr,    \
                                                              bsr_col_ind,    \
                                                              row_block_dim,  \
                                                              col_block_dim,  \
                                                              x,              \
                                                              beta,           \
                                                              y));            \
        return rocsparse_status_success;                                      \
    }                                                                         \
    catch(...)                                                                \
    {                                                                         \
        RETURN_ROCSPARSE_EXCEPTION();                                         \
    }

C_IMPL(rocsparse_sgebsrmv, float);
C_IMPL(rocsparse_dgebsrmv, double);
C_IMPL(rocsparse_cgebsrmv, rocsparse_float_complex);
C_IMPL(rocsparse_zgebsrmv, rocsparse_double_complex);

#undef C_IMPL

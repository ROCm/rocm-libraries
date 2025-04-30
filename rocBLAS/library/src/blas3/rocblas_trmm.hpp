/* ************************************************************************
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include "check_numerics_matrix.hpp"
#include "definitions.hpp"
#include "rocblas_gemm.hpp"

template <rocblas_int DIM_X, rocblas_int DIM_Y, typename TScal, typename TPtr>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
set_matrix_zero_if_alpha_zero_kernel(rocblas_int    m,
                                     rocblas_int    n,
                                     TScal          alpha_device_host,
                                     rocblas_stride stride_alpha,
                                     TPtr           Aa,
                                     rocblas_int    lda,
                                     rocblas_stride a_st_or_of);

template <typename TScal, typename TPtr>
rocblas_status rocblas_set_matrix_zero_if_alpha_zero_template(rocblas_handle handle,
                                                              rocblas_int    m,
                                                              rocblas_int    n,
                                                              TScal          alpha,
                                                              rocblas_stride stride_alpha,
                                                              TPtr           A,
                                                              int64_t        lda,
                                                              rocblas_stride a_st_or_of,
                                                              rocblas_int    batch_count);

template <typename API_INT, typename TScal, typename TPtr, typename TConstPtr>
rocblas_status rocblas_trmm_arg_check(rocblas_handle    handle,
                                      rocblas_side      side,
                                      rocblas_fill      uplo,
                                      rocblas_operation trans,
                                      rocblas_diagonal  diag,
                                      API_INT           m,
                                      API_INT           n,
                                      const TScal*      alpha,
                                      TConstPtr         a,
                                      API_INT           lda,
                                      TConstPtr         b,
                                      API_INT           ldb,
                                      TPtr              c,
                                      API_INT           ldc,
                                      API_INT           batch_count)
{
    if(side != rocblas_side_left && side != rocblas_side_right)
        return rocblas_status_invalid_value;

    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;

    if(trans != rocblas_operation_none && trans != rocblas_operation_transpose
       && trans != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;

    if(diag != rocblas_diagonal_non_unit && diag != rocblas_diagonal_unit)
        return rocblas_status_invalid_value;

    if(batch_count < 0 || m < 0 || n < 0 || ldc < m || ldb < m
       || (side == rocblas_side_left && (lda < m)) || (side != rocblas_side_left && (lda < n)))
        return rocblas_status_invalid_size;

    if(!m || !n || !batch_count)
        return rocblas_status_success;

    if(!c || !alpha
       || (handle->pointer_mode == rocblas_pointer_mode_host && *alpha != 0 && (!a || !b)))
        return rocblas_status_invalid_pointer;

    // ensuring ldb == ldc when B and C are the same
    // matching gemm_ex behaviour
    if(b == c && ldb != ldc)
        return rocblas_status_invalid_value;

    return rocblas_status_continue;
}

template <int NB, bool BATCHED, typename T, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_internal_trmm_launcher(rocblas_handle    handle,
                                              rocblas_side      side,
                                              rocblas_fill      uplo,
                                              rocblas_operation trans_a,
                                              rocblas_diagonal  diag,
                                              rocblas_int       m,
                                              rocblas_int       n,
                                              TScal*            alpha,
                                              rocblas_stride    stride_alpha,
                                              TConstPtr*        dA,
                                              rocblas_stride    offset_a,
                                              int64_t           lda,
                                              rocblas_stride    stride_a,
                                              TConstPtr*        dB,
                                              rocblas_stride    offset_b,
                                              int64_t           ldb,
                                              rocblas_stride    stride_b,
                                              TPtr*             dC,
                                              rocblas_stride    offset_c,
                                              int64_t           ldc,
                                              rocblas_stride    stride_c,
                                              rocblas_int       batch_count);

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_trmm_template(rocblas_handle    handle,
                                   rocblas_side      side,
                                   rocblas_fill      uplo,
                                   rocblas_operation trans_a,
                                   rocblas_diagonal  diag,
                                   rocblas_int       m,
                                   rocblas_int       n,
                                   const T*          alpha,
                                   rocblas_stride    stride_alpha,
                                   const T*          dA,
                                   rocblas_stride    offset_a,
                                   rocblas_int       lda,
                                   rocblas_stride    stride_a,
                                   const T*          dB,
                                   rocblas_stride    offset_b,
                                   rocblas_int       ldb,
                                   rocblas_stride    stride_b,
                                   T*                dC,
                                   rocblas_stride    offset_c,
                                   rocblas_int       lddc,
                                   rocblas_stride    stride_c,
                                   rocblas_int       batch_count);

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_trmm_batched_template(rocblas_handle    handle,
                                           rocblas_side      side,
                                           rocblas_fill      uplo,
                                           rocblas_operation trans_a,
                                           rocblas_diagonal  diag,
                                           rocblas_int       m,
                                           rocblas_int       n,
                                           const T*          alpha,
                                           rocblas_stride    stride_alpha,
                                           const T* const*   dA,
                                           rocblas_stride    offset_a,
                                           rocblas_int       lda,
                                           rocblas_stride    stride_a,
                                           const T* const*   dB,
                                           rocblas_stride    offset_b,
                                           rocblas_int       ldb,
                                           rocblas_stride    stride_b,
                                           T* const*         dC,
                                           rocblas_stride    offset_c,
                                           rocblas_int       lddc,
                                           rocblas_stride    stride_c,
                                           rocblas_int       batch_count);

template <typename TConstPtr, typename TPtr>
rocblas_status rocblas_trmm_check_numerics(const char*       function_name,
                                           rocblas_handle    handle,
                                           rocblas_side      side,
                                           rocblas_fill      uplo,
                                           rocblas_operation trans_a,
                                           int64_t           m,
                                           int64_t           n,
                                           TConstPtr*        A,
                                           int64_t           lda,
                                           rocblas_stride    stride_a,
                                           TPtr*             B,
                                           int64_t           ldb,
                                           rocblas_stride    stride_b,
                                           int64_t           batch_count,
                                           const int         check_numerics,
                                           bool              is_input);

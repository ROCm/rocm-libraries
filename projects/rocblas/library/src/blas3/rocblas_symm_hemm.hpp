/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "handle.hpp"

template <typename API_INT, typename TScal, typename TConstPtr, typename TPtr>
inline rocblas_status rocblas_symm_arg_check(rocblas_handle handle,
                                             rocblas_side   side,
                                             rocblas_fill   uplo,
                                             API_INT        m,
                                             API_INT        n,
                                             TScal          alpha,
                                             TConstPtr      AP,
                                             rocblas_stride offsetA,
                                             API_INT        lda,
                                             rocblas_stride strideA,
                                             TConstPtr      BP,
                                             rocblas_stride offsetB,
                                             API_INT        ldb,
                                             rocblas_stride strideB,
                                             TScal          beta,
                                             const TPtr     CP,
                                             rocblas_stride offsetC,
                                             API_INT        ldc,
                                             rocblas_stride strideC,
                                             API_INT        batch_count)
{
    if(side != rocblas_side_left && side != rocblas_side_right)
        return rocblas_status_invalid_value;

    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;

    if(batch_count < 0 || m < 0 || n < 0 || ldc < m || ldb < m
       || (side == rocblas_side_left && (lda < m)) || (side != rocblas_side_left && (lda < n)))
        return rocblas_status_invalid_size;

    if(!m || !n || !batch_count)
        return rocblas_status_success;

    if(!beta || !alpha)
        return rocblas_status_invalid_pointer;

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        if(*alpha == 0 && *beta == 1)
            return rocblas_status_success;

        if(!CP || (*alpha != 0 && (!AP || !BP)))
            return rocblas_status_invalid_pointer;
    }

    return rocblas_status_continue;
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_symm_template(rocblas_handle handle,
                                   rocblas_side   side,
                                   rocblas_fill   uplo,
                                   rocblas_int    m,
                                   rocblas_int    n,
                                   const T*       alpha,
                                   const T*       A,
                                   rocblas_stride offsetA,
                                   rocblas_int    lda,
                                   rocblas_stride strideA,
                                   const T*       B,
                                   rocblas_stride offsetB,
                                   rocblas_int    ldb,
                                   rocblas_stride strideB,
                                   const T*       beta,
                                   T*             C,
                                   rocblas_stride offsetC,
                                   rocblas_int    ldc,
                                   rocblas_stride strideC,
                                   rocblas_int    batch_count);

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_symm_batched_template(rocblas_handle  handle,
                                           rocblas_side    side,
                                           rocblas_fill    uplo,
                                           rocblas_int     m,
                                           rocblas_int     n,
                                           const T*        alpha,
                                           const T* const* A,
                                           rocblas_stride  offsetA,
                                           rocblas_int     lda,
                                           rocblas_stride  strideA,
                                           const T* const* B,
                                           rocblas_stride  offsetB,
                                           rocblas_int     ldb,
                                           rocblas_stride  strideB,
                                           const T*        beta,
                                           T* const*       C,
                                           rocblas_stride  offsetC,
                                           rocblas_int     ldc,
                                           rocblas_stride  strideC,
                                           rocblas_int     batch_count);

template <bool HERM, typename T>
rocblas_status rocblas_internal_symm_hemm_launcher(rocblas_handle handle,
                                                   rocblas_side   side,
                                                   rocblas_fill   uplo,
                                                   rocblas_int    m,
                                                   rocblas_int    n,
                                                   const T*       alpha,
                                                   const T*       A,
                                                   rocblas_stride offsetA,
                                                   int64_t        lda,
                                                   rocblas_stride strideA,
                                                   const T*       B,
                                                   rocblas_stride offsetB,
                                                   int64_t        ldb,
                                                   rocblas_stride strideB,
                                                   const T*       beta,
                                                   T*             C,
                                                   rocblas_stride offsetC,
                                                   int64_t        ldc,
                                                   rocblas_stride strideC,
                                                   rocblas_int    batch_count);

template <bool HERM, typename T>
rocblas_status rocblas_internal_symm_hemm_batched_launcher(rocblas_handle  handle,
                                                           rocblas_side    side,
                                                           rocblas_fill    uplo,
                                                           rocblas_int     m,
                                                           rocblas_int     n,
                                                           const T*        alpha,
                                                           const T* const* A,
                                                           rocblas_stride  offsetA,
                                                           int64_t         lda,
                                                           rocblas_stride  strideA,
                                                           const T* const* B,
                                                           rocblas_stride  offsetB,
                                                           int64_t         ldb,
                                                           rocblas_stride  strideB,
                                                           const T*        beta,
                                                           T* const*       C,
                                                           rocblas_stride  offsetC,
                                                           int64_t         ldc,
                                                           rocblas_stride  strideC,
                                                           rocblas_int     batch_count);

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_hemm_template(rocblas_handle handle,
                                   rocblas_side   side,
                                   rocblas_fill   uplo,
                                   rocblas_int    m,
                                   rocblas_int    n,
                                   const T*       alpha,
                                   const T*       A,
                                   rocblas_stride offsetA,
                                   rocblas_int    lda,
                                   rocblas_stride strideA,
                                   const T*       B,
                                   rocblas_stride offsetB,
                                   rocblas_int    ldb,
                                   rocblas_stride strideB,
                                   const T*       beta,
                                   T*             C,
                                   rocblas_stride offsetC,
                                   rocblas_int    ldc,
                                   rocblas_stride strideC,
                                   rocblas_int    batch_count);

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_hemm_batched_template(rocblas_handle  handle,
                                           rocblas_side    side,
                                           rocblas_fill    uplo,
                                           rocblas_int     m,
                                           rocblas_int     n,
                                           const T*        alpha,
                                           const T* const* A,
                                           rocblas_stride  offsetA,
                                           rocblas_int     lda,
                                           rocblas_stride  strideA,
                                           const T* const* B,
                                           rocblas_stride  offsetB,
                                           rocblas_int     ldb,
                                           rocblas_stride  strideB,
                                           const T*        beta,
                                           T* const*       C,
                                           rocblas_stride  offsetC,
                                           rocblas_int     ldc,
                                           rocblas_stride  strideC,
                                           rocblas_int     batch_count);

template <bool HERM, typename TConstPtr, typename TPtr>
rocblas_status rocblas_hemm_symm_check_numerics(const char*    function_name,
                                                rocblas_handle handle,
                                                rocblas_side   side,
                                                rocblas_fill   uplo,
                                                int64_t        m,
                                                int64_t        n,
                                                TConstPtr      A,
                                                int64_t        lda,
                                                rocblas_stride strideA,
                                                TConstPtr      B,
                                                int64_t        ldb,
                                                rocblas_stride strideB,
                                                TPtr           C,
                                                int64_t        ldc,
                                                rocblas_stride strideC,
                                                int64_t        batch_count,
                                                const int      check_numerics,
                                                bool           is_input);

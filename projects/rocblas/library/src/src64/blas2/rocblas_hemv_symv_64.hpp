/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "handle.hpp"

template <bool IS_HEMV, typename T, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_internal_symv_hemv_launcher_64(rocblas_handle handle,
                                                      rocblas_fill   uplo,
                                                      int64_t        n_64,
                                                      TScal          alpha,
                                                      rocblas_stride stride_alpha,
                                                      TConstPtr      A,
                                                      rocblas_stride offsetA,
                                                      int64_t        lda,
                                                      rocblas_stride strideA,
                                                      TConstPtr      x,
                                                      rocblas_stride offsetx,
                                                      int64_t        incx,
                                                      rocblas_stride stridex,
                                                      TScal          beta,
                                                      rocblas_stride stride_beta,
                                                      TPtr           y,
                                                      rocblas_stride offsety,
                                                      int64_t        incy,
                                                      rocblas_stride stridey,
                                                      int64_t        batch_count_64,
                                                      T*             workspace);

template <typename T>
rocblas_status rocblas_internal_symv_template_64(rocblas_handle handle,
                                                 rocblas_fill   uplo,
                                                 int64_t        n_64,
                                                 const T*       alpha,
                                                 rocblas_stride stride_alpha,
                                                 const T*       A,
                                                 rocblas_stride offsetA,
                                                 int64_t        lda,
                                                 rocblas_stride strideA,
                                                 const T*       x,
                                                 rocblas_stride offsetx,
                                                 int64_t        incx,
                                                 rocblas_stride stridex,
                                                 const T*       beta,
                                                 rocblas_stride stride_beta,
                                                 T*             y,
                                                 rocblas_stride offsety,
                                                 int64_t        incy,
                                                 rocblas_stride stridey,
                                                 int64_t        batch_count_64,
                                                 T*             workspace);

template <typename T>
rocblas_status rocblas_internal_hemv_template_64(rocblas_handle handle,
                                                 rocblas_fill   uplo,
                                                 int64_t        n_64,
                                                 const T*       alpha,
                                                 rocblas_stride stride_alpha,
                                                 const T*       A,
                                                 rocblas_stride offsetA,
                                                 int64_t        lda,
                                                 rocblas_stride strideA,
                                                 const T*       x,
                                                 rocblas_stride offsetx,
                                                 int64_t        incx,
                                                 rocblas_stride stridex,
                                                 const T*       beta,
                                                 rocblas_stride stride_beta,
                                                 T*             y,
                                                 rocblas_stride offsety,
                                                 int64_t        incy,
                                                 rocblas_stride stridey,
                                                 int64_t        batch_count_64,
                                                 T*             workspace);

template <typename T>
rocblas_status rocblas_internal_symv_batched_template_64(rocblas_handle  handle,
                                                         rocblas_fill    uplo,
                                                         int64_t         n_64,
                                                         const T*        alpha,
                                                         rocblas_stride  stride_alpha,
                                                         const T* const* A,
                                                         rocblas_stride  offsetA,
                                                         int64_t         lda,
                                                         rocblas_stride  strideA,
                                                         const T* const* x,
                                                         rocblas_stride  offsetx,
                                                         int64_t         incx,
                                                         rocblas_stride  stridex,
                                                         const T*        beta,
                                                         rocblas_stride  stride_beta,
                                                         T* const*       y,
                                                         rocblas_stride  offsety,
                                                         int64_t         incy,
                                                         rocblas_stride  stridey,
                                                         int64_t         batch_count_64,
                                                         T*              workspace);

template <typename T>
rocblas_status rocblas_internal_hemv_batched_template_64(rocblas_handle  handle,
                                                         rocblas_fill    uplo,
                                                         int64_t         n_64,
                                                         const T*        alpha,
                                                         rocblas_stride  stride_alpha,
                                                         const T* const* A,
                                                         rocblas_stride  offsetA,
                                                         int64_t         lda,
                                                         rocblas_stride  strideA,
                                                         const T* const* x,
                                                         rocblas_stride  offsetx,
                                                         int64_t         incx,
                                                         rocblas_stride  stridex,
                                                         const T*        beta,
                                                         rocblas_stride  stride_beta,
                                                         T* const*       y,
                                                         rocblas_stride  offsety,
                                                         int64_t         incy,
                                                         rocblas_stride  stridey,
                                                         int64_t         batch_count_64,
                                                         T*              workspace);

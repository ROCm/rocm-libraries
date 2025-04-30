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

template <bool BATCHED, typename T, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_internal_trmm_launcher_64(rocblas_handle    handle,
                                                 rocblas_side      side,
                                                 rocblas_fill      uplo,
                                                 rocblas_operation trans_a,
                                                 rocblas_diagonal  diag,
                                                 int64_t           m_64,
                                                 int64_t           n_64,
                                                 TScal*            alpha,
                                                 rocblas_stride    stride_alpha,
                                                 TConstPtr*        dA,
                                                 rocblas_stride    offset_a,
                                                 int64_t           lda_64,
                                                 rocblas_stride    stride_a,
                                                 TConstPtr*        dB,
                                                 rocblas_stride    offset_b,
                                                 int64_t           ldb_64,
                                                 rocblas_stride    stride_b,
                                                 TPtr*             dC,
                                                 rocblas_stride    offset_c,
                                                 int64_t           ldc_64,
                                                 rocblas_stride    stride_c,
                                                 int64_t           batch_count);

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_trmm_template_64(rocblas_handle    handle,
                                      rocblas_side      side,
                                      rocblas_fill      uplo,
                                      rocblas_operation trans_a,
                                      rocblas_diagonal  diag,
                                      int64_t           m_64,
                                      int64_t           n_64,
                                      const T*          alpha,
                                      rocblas_stride    stride_alpha,
                                      const T*          dA,
                                      rocblas_stride    offset_a,
                                      int64_t           lda_64,
                                      rocblas_stride    stride_a,
                                      const T*          dB,
                                      rocblas_stride    offset_b,
                                      int64_t           ldb_64,
                                      rocblas_stride    stride_b,
                                      T*                dC,
                                      rocblas_stride    offset_c,
                                      int64_t           ldc_64,
                                      rocblas_stride    stride_c,
                                      int64_t           batch_count);

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_trmm_batched_template_64(rocblas_handle    handle,
                                              rocblas_side      side,
                                              rocblas_fill      uplo,
                                              rocblas_operation trans_a,
                                              rocblas_diagonal  diag,
                                              int64_t           m_64,
                                              int64_t           n_64,
                                              const T*          alpha,
                                              rocblas_stride    stride_alpha,
                                              const T* const*   dA,
                                              rocblas_stride    offset_a,
                                              int64_t           lda_64,
                                              rocblas_stride    stride_a,
                                              const T* const*   dB,
                                              rocblas_stride    offset_b,
                                              int64_t           ldb_64,
                                              rocblas_stride    stride_b,
                                              T* const*         dC,
                                              rocblas_stride    offset_c,
                                              int64_t           ldc_64,
                                              rocblas_stride    stride_c,
                                              int64_t           batch_count);

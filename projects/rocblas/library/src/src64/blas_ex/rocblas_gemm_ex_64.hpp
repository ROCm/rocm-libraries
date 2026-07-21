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

#include "int64_helpers.hpp"

template <bool BATCHED>
rocblas_status rocblas_gemm_ex_template_64(rocblas_handle    handle,
                                           rocblas_operation trans_a,
                                           rocblas_operation trans_b,
                                           int64_t           m,
                                           int64_t           n,
                                           int64_t           k,
                                           const void*       alpha,
                                           const void*       a,
                                           rocblas_datatype  a_type,
                                           rocblas_stride    offsetAin,
                                           int64_t           lda,
                                           rocblas_stride    stride_a,
                                           const void*       b,
                                           rocblas_datatype  b_type,
                                           rocblas_stride    offsetBin,
                                           int64_t           ldb,
                                           rocblas_stride    stride_b,
                                           const void*       beta,
                                           const void*       c,
                                           rocblas_datatype  c_type,
                                           rocblas_stride    offsetCin,
                                           int64_t           ldc,
                                           rocblas_stride    stride_c,
                                           void*             d,
                                           rocblas_datatype  d_type,
                                           rocblas_stride    offsetDin,
                                           int64_t           ldd,
                                           rocblas_stride    stride_d,
                                           int64_t           batch_count,
                                           rocblas_datatype  compute_type,
                                           rocblas_gemm_algo algo,
                                           int32_t           solution_index,
                                           uint32_t          flags);

template <bool BATCHED, typename Ti, typename To = Ti, typename TScal = To>
rocblas_status rocblas_internal_gemm_ex_typecasting_64(rocblas_handle     handle,
                                                       rocblas_operation  trans_a,
                                                       rocblas_operation  trans_b,
                                                       int64_t            m_64,
                                                       int64_t            n_64,
                                                       int64_t            k_64,
                                                       const void*        alpha,
                                                       const void*        a,
                                                       rocblas_stride     offsetAin,
                                                       int64_t            lda_64,
                                                       rocblas_stride     stride_a,
                                                       const void*        b,
                                                       rocblas_stride     offsetBin,
                                                       int64_t            ldb_64,
                                                       rocblas_stride     stride_b,
                                                       const void*        beta,
                                                       const void*        c,
                                                       rocblas_stride     offsetCin,
                                                       int64_t            ldc_64,
                                                       rocblas_stride     stride_c,
                                                       void*              d,
                                                       rocblas_stride     offsetDin,
                                                       int64_t            ldd_64,
                                                       rocblas_stride     stride_d,
                                                       int64_t            batch_count_64,
                                                       rocblas_gemm_algo  algo,
                                                       int32_t            solution_index,
                                                       rocblas_gemm_flags flags);

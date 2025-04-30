/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "blas3/rocblas_geam.hpp"

/**
 * TScal     is always: const T* (either host or device)
 * TConstPtr is either: const T* OR const T* const*
 * TPtr      is either:       T* OR       T* const*
 * Where T is the base type (float, double, rocblas_complex, or rocblas_double_complex)
 */

template <typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_geam_launcher_64(rocblas_handle    handle,
                                        rocblas_operation transA,
                                        rocblas_operation transB,
                                        int64_t           m_64,
                                        int64_t           n_64,
                                        TScal             alpha,
                                        TConstPtr         A,
                                        rocblas_stride    offset_a,
                                        int64_t           lda_64,
                                        rocblas_stride    stride_a,
                                        TScal             beta,
                                        TConstPtr         B,
                                        rocblas_stride    offset_b,
                                        int64_t           ldb_64,
                                        rocblas_stride    stride_b,
                                        TPtr              C,
                                        rocblas_stride    offset_c,
                                        int64_t           ldc_64,
                                        rocblas_stride    stride_c,
                                        int64_t           batch_count_64);


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

#include "rocblas_gemmt_kernels.hpp"
#include "../blas3/rocblas_gemm.hpp"
#include "definitions.hpp"
#include "handle.hpp"
#include "rocblas_blas_ex_threshold.hpp"
#include "rocblas_block_sizes.h"
#include "rocblas_gemmt.hpp"
#include "utility.hpp"

#ifdef INSTANTIATE_GEMMT_LAUNCHER
#error INSTANTIATE_GEMMT_LAUNCHER already defined
#endif

#define INSTANTIATE_GEMMT_LAUNCHER(API_INT_, TScal_, TConstPtr_, TPtr_)                           \
    template rocblas_status rocblas_internal_gemmt_launcher<API_INT_, TScal_, TConstPtr_, TPtr_>( \
        rocblas_handle    handle,                                                                 \
        rocblas_fill      uplo,                                                                   \
        rocblas_operation transA,                                                                 \
        rocblas_operation transB,                                                                 \
        rocblas_int       n,                                                                      \
        API_INT_          k,                                                                      \
        const TScal_*     alpha,                                                                  \
        TConstPtr_        dA_in,                                                                  \
        API_INT_          lda,                                                                    \
        rocblas_stride    stride_a,                                                               \
        TConstPtr_        dB_in,                                                                  \
        API_INT_          ldb,                                                                    \
        rocblas_stride    stride_b,                                                               \
        const TScal_*     beta,                                                                   \
        TPtr_             dC_in,                                                                  \
        API_INT_          ldc,                                                                    \
        rocblas_stride    stride_c,                                                               \
        rocblas_int       batch_count);

// non batched

INSTANTIATE_GEMMT_LAUNCHER(rocblas_int, float, const float*, float*)
INSTANTIATE_GEMMT_LAUNCHER(rocblas_int, double, const double*, double*)
INSTANTIATE_GEMMT_LAUNCHER(rocblas_int,
                           rocblas_float_complex,
                           const rocblas_float_complex*,
                           rocblas_float_complex*)
INSTANTIATE_GEMMT_LAUNCHER(rocblas_int,
                           rocblas_double_complex,
                           const rocblas_double_complex*,
                           rocblas_double_complex*)

INSTANTIATE_GEMMT_LAUNCHER(int64_t, float, const float*, float*)
INSTANTIATE_GEMMT_LAUNCHER(int64_t, double, const double*, double*)
INSTANTIATE_GEMMT_LAUNCHER(int64_t,
                           rocblas_float_complex,
                           const rocblas_float_complex*,
                           rocblas_float_complex*)
INSTANTIATE_GEMMT_LAUNCHER(int64_t,
                           rocblas_double_complex,
                           const rocblas_double_complex*,
                           rocblas_double_complex*)
// batched

INSTANTIATE_GEMMT_LAUNCHER(rocblas_int, float, const float* const*, float* const*)
INSTANTIATE_GEMMT_LAUNCHER(rocblas_int, double, const double* const*, double* const*)
INSTANTIATE_GEMMT_LAUNCHER(rocblas_int,
                           rocblas_float_complex,
                           const rocblas_float_complex* const*,
                           rocblas_float_complex* const*)
INSTANTIATE_GEMMT_LAUNCHER(rocblas_int,
                           rocblas_double_complex,
                           const rocblas_double_complex* const*,
                           rocblas_double_complex* const*)

INSTANTIATE_GEMMT_LAUNCHER(int64_t, float, const float* const*, float* const*)
INSTANTIATE_GEMMT_LAUNCHER(int64_t, double, const double* const*, double* const*)
INSTANTIATE_GEMMT_LAUNCHER(int64_t,
                           rocblas_float_complex,
                           const rocblas_float_complex* const*,
                           rocblas_float_complex* const*)
INSTANTIATE_GEMMT_LAUNCHER(int64_t,
                           rocblas_double_complex,
                           const rocblas_double_complex* const*,
                           rocblas_double_complex* const*)

#undef INSTANTIATE_GEMMT_LAUNCHER

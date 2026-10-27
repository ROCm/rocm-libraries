/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

// Instantiations in this file are for the (array-of-pointers) batched trsv API,
// which calls through rocblas_internal_trsv_batched_template. The non-batched
// and strided-batched instantiations live in rocblas_trsv_kernels.cpp, since
// those two APIs share rocblas_internal_trsv_template.

#include "rocblas_trsv_kernels.hpp"

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files *trsv*.cpp

// clang-format off

INSTANTIATE_TRSV_NUMERICS(float const* const*, float* const*)
INSTANTIATE_TRSV_NUMERICS(double const* const*, double* const*)
INSTANTIATE_TRSV_NUMERICS(rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_TRSV_NUMERICS(rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_TRSV_NUMERICS

#ifdef INSTANTIATE_TRSV_BATCHED_TEMPLATE
#error INSTANTIATE_TRSV_BATCHED_TEMPLATE already defined
#endif

#define INSTANTIATE_TRSV_BATCHED_TEMPLATE(T_)                                                       \
template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status rocblas_internal_trsv_batched_template<T_> \
                                               (rocblas_handle    handle,                           \
                                                rocblas_fill      uplo,                             \
                                                rocblas_operation transA,                           \
                                                rocblas_diagonal  diag,                             \
                                                rocblas_int       n,                                \
                                                const T_* const*  dA,                               \
                                                rocblas_stride    offset_A,                         \
                                                rocblas_int       lda,                              \
                                                rocblas_stride    stride_A,                         \
                                                T_* const*        dx,                               \
                                                rocblas_stride    offset_x,                         \
                                                rocblas_int       incx,                             \
                                                rocblas_stride    stride_x,                         \
                                                rocblas_int       batch_count,                      \
                                                rocblas_int*      w_completed_sec);



INSTANTIATE_TRSV_BATCHED_TEMPLATE(float)
INSTANTIATE_TRSV_BATCHED_TEMPLATE(double)
INSTANTIATE_TRSV_BATCHED_TEMPLATE(rocblas_float_complex)
INSTANTIATE_TRSV_BATCHED_TEMPLATE(rocblas_double_complex)

#undef INSTANTIATE_TRSV_BATCHED_TEMPLATE

// clang-format on

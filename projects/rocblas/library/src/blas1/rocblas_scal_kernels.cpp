/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "handle.hpp"
#include "rocblas.h"
#include "rocblas_block_sizes.h"

#include "blas1/rocblas_scal.hpp"
#include "blas1/rocblas_scal_kernels.hpp"

template <typename T, typename Ta>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_scal_template(rocblas_handle handle,
                                   rocblas_int    n,
                                   const Ta*      alpha,
                                   rocblas_stride stride_alpha,
                                   T*             x,
                                   rocblas_stride offset_x,
                                   rocblas_int    incx,
                                   rocblas_stride stride_x,
                                   rocblas_int    batch_count)
{
    return rocblas_internal_scal_launcher<rocblas_int, ROCBLAS_SCAL_NB, T, T>(
        handle, n, alpha, stride_alpha, x, offset_x, incx, stride_x, batch_count);
}

template <typename T, typename Ta>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_scal_batched_template(rocblas_handle handle,
                                           rocblas_int    n,
                                           const Ta*      alpha,
                                           rocblas_stride stride_alpha,
                                           T* const*      x,
                                           rocblas_stride offset_x,
                                           rocblas_int    incx,
                                           rocblas_stride stride_x,
                                           rocblas_int    batch_count)
{
    return rocblas_internal_scal_launcher<rocblas_int, ROCBLAS_SCAL_NB, T, T>(
        handle, n, alpha, stride_alpha, x, offset_x, incx, stride_x, batch_count);
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files scal*.cpp

// clang-format off
#ifdef INSTANTIATE_SCAL_TEMPLATE
#error INSTANTIATE_SCAL_TEMPLATE already defined
#endif

#define INSTANTIATE_SCAL_TEMPLATE(T_, Ta_)                              \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status            \
        rocblas_internal_scal_template<T_, Ta_>(rocblas_handle handle,  \
                                           rocblas_int    n,            \
                                           const Ta_*     alpha,        \
                                           rocblas_stride stride_alpha, \
                                           T_*            x,            \
                                           rocblas_stride offset_x,     \
                                           rocblas_int    incx,         \
                                           rocblas_stride stride_x,     \
                                           rocblas_int    batch_count);

// Not exporting execution type
INSTANTIATE_SCAL_TEMPLATE(rocblas_half, rocblas_half)
INSTANTIATE_SCAL_TEMPLATE(rocblas_half, float)
INSTANTIATE_SCAL_TEMPLATE(float, float)
INSTANTIATE_SCAL_TEMPLATE(double, double)
INSTANTIATE_SCAL_TEMPLATE(rocblas_float_complex, rocblas_float_complex)
INSTANTIATE_SCAL_TEMPLATE(rocblas_double_complex, rocblas_double_complex)
INSTANTIATE_SCAL_TEMPLATE(rocblas_float_complex, float)
INSTANTIATE_SCAL_TEMPLATE(rocblas_double_complex, double)

#undef INSTANTIATE_SCAL_TEMPLATE

#ifdef INSTANTIATE_SCAL_BATCHED_TEMPLATE
#error INSTANTIATE_SCAL_BATCHED_TEMPLATE already defined
#endif

#define INSTANTIATE_SCAL_BATCHED_TEMPLATE(T_, Ta_)                              \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                    \
        rocblas_internal_scal_batched_template<T_, Ta_>(rocblas_handle handle,  \
                                                   rocblas_int    n,            \
                                                   const Ta_*     alpha,        \
                                                   rocblas_stride stride_alpha, \
                                                   T_* const*     x,            \
                                                   rocblas_stride offset_x,     \
                                                   rocblas_int    incx,         \
                                                   rocblas_stride stride_x,     \
                                                   rocblas_int    batch_count);

INSTANTIATE_SCAL_BATCHED_TEMPLATE(rocblas_half, rocblas_half)
INSTANTIATE_SCAL_BATCHED_TEMPLATE(rocblas_half, float)
INSTANTIATE_SCAL_BATCHED_TEMPLATE(float, float)
INSTANTIATE_SCAL_BATCHED_TEMPLATE(double, double)
INSTANTIATE_SCAL_BATCHED_TEMPLATE(rocblas_float_complex, rocblas_float_complex)
INSTANTIATE_SCAL_BATCHED_TEMPLATE(rocblas_double_complex, rocblas_double_complex)
INSTANTIATE_SCAL_BATCHED_TEMPLATE(rocblas_float_complex, float)
INSTANTIATE_SCAL_BATCHED_TEMPLATE(rocblas_double_complex, double)

#undef INSTANTIATE_SCAL_BATCHED_TEMPLATE

#ifdef INST_SCAL_EX_LAUNCHER
#error INST_SCAL_EX_LAUNCHER already defined
#endif

#define INST_SCAL_EX_LAUNCHER(NB_, T_, Tex_, Ta_, Tx_)                                \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                                 \
        rocblas_internal_scal_launcher<rocblas_int, NB_, T_, Tex_, Ta_, Tx_>(rocblas_handle handle,       \
                                                                rocblas_int    n,            \
                                                                const Ta_*     alpha,        \
                                                                rocblas_stride stride_alpha, \
                                                                Tx_            x,            \
                                                                rocblas_stride offset_x,     \
                                                                rocblas_int    incx,         \
                                                                rocblas_stride stride_x,     \
                                                                rocblas_int    batch_count);

// Instantiations for scal_ex
INST_SCAL_EX_LAUNCHER(ROCBLAS_SCAL_NB, rocblas_half, float, rocblas_half, rocblas_half*)
INST_SCAL_EX_LAUNCHER(ROCBLAS_SCAL_NB, rocblas_half, float, float, rocblas_half*)
INST_SCAL_EX_LAUNCHER(ROCBLAS_SCAL_NB, rocblas_bfloat16, float, rocblas_bfloat16, rocblas_bfloat16*)
INST_SCAL_EX_LAUNCHER(ROCBLAS_SCAL_NB, rocblas_bfloat16, float, float, rocblas_bfloat16*)

INST_SCAL_EX_LAUNCHER(ROCBLAS_SCAL_NB, rocblas_half, float, rocblas_half, rocblas_half* const*)
INST_SCAL_EX_LAUNCHER(ROCBLAS_SCAL_NB, rocblas_half, float, float, rocblas_half* const*)
INST_SCAL_EX_LAUNCHER(ROCBLAS_SCAL_NB, rocblas_bfloat16, float, rocblas_bfloat16, rocblas_bfloat16* const*)
INST_SCAL_EX_LAUNCHER(ROCBLAS_SCAL_NB, rocblas_bfloat16, float, float, rocblas_bfloat16* const*)

#undef INST_SCAL_EX_LAUNCHER

// clang-format on

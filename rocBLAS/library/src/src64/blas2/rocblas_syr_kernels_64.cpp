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

#include "handle.hpp"
#include "int64_helpers.hpp"

#include "rocblas_syr_64.hpp"

#include "blas2/rocblas_syr.hpp" // int32 API called

template <typename T, typename U, typename V, typename W>
rocblas_status rocblas_internal_syr_launcher_64(rocblas_handle handle,
                                                rocblas_fill   uplo,
                                                int64_t        n_64,
                                                U              alpha,
                                                rocblas_stride stride_alpha,
                                                V              x,
                                                rocblas_stride offset_x,
                                                int64_t        incx_64,
                                                rocblas_stride stride_x,
                                                W              A,
                                                rocblas_stride offset_A,
                                                int64_t        lda_64,
                                                rocblas_stride stride_A,
                                                int64_t        batch_count_64)
{
    // Quick return if possible. Not Argument error
    if(!n_64 || !batch_count_64)
        return rocblas_status_success;

    if(n_64 > c_i32_max)
        return rocblas_status_invalid_size; // defer adding new kernels for sizes exceeding practical memory

    for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
    {
        auto    x_ptr       = adjust_ptr_batch(x, b_base, stride_x);
        auto    A_ptr       = adjust_ptr_batch(A, b_base, stride_A);
        int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

        rocblas_status status = rocblas_internal_syr_launcher<T>(handle,
                                                                 uplo,
                                                                 (rocblas_int)n_64,
                                                                 alpha,
                                                                 stride_alpha,
                                                                 x_ptr,
                                                                 offset_x,
                                                                 incx_64,
                                                                 stride_x,
                                                                 A_ptr,
                                                                 offset_A,
                                                                 lda_64,
                                                                 stride_A,
                                                                 batch_count);

        if(status != rocblas_status_success)
            return status;
    } // batch
    return rocblas_status_success;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files *syr*.cpp

#ifdef INSTANTIATE_SYR_LAUNCHER_64
#error INSTANTIATE_SYR_LAUNCHER_64 already defined
#endif

#define INSTANTIATE_SYR_LAUNCHER_64(T_, U_, V_, W_)                           \
    template rocblas_status rocblas_internal_syr_launcher_64<T_, U_, V_, W_>( \
        rocblas_handle handle,                                                \
        rocblas_fill   uplo,                                                  \
        int64_t        n_64,                                                  \
        U_             alpha,                                                 \
        rocblas_stride stride_alpha,                                          \
        V_             x,                                                     \
        rocblas_stride offsetx,                                               \
        int64_t        incx,                                                  \
        rocblas_stride stridex,                                               \
        W_             A,                                                     \
        rocblas_stride offseta,                                               \
        int64_t        lda,                                                   \
        rocblas_stride strideA,                                               \
        int64_t        batch_count_64);

INSTANTIATE_SYR_LAUNCHER_64(float, float const*, float const*, float*);
INSTANTIATE_SYR_LAUNCHER_64(double, double const*, double const*, double*);
INSTANTIATE_SYR_LAUNCHER_64(rocblas_float_complex,
                            rocblas_float_complex const*,
                            rocblas_float_complex const*,
                            rocblas_float_complex*);
INSTANTIATE_SYR_LAUNCHER_64(rocblas_double_complex,
                            rocblas_double_complex const*,
                            rocblas_double_complex const*,
                            rocblas_double_complex*);
INSTANTIATE_SYR_LAUNCHER_64(float, float const*, float const* const*, float* const*);
INSTANTIATE_SYR_LAUNCHER_64(double, double const*, double const* const*, double* const*);
INSTANTIATE_SYR_LAUNCHER_64(rocblas_float_complex,
                            rocblas_float_complex const*,
                            rocblas_float_complex const* const*,
                            rocblas_float_complex* const*);
INSTANTIATE_SYR_LAUNCHER_64(rocblas_double_complex,
                            rocblas_double_complex const*,
                            rocblas_double_complex const* const*,
                            rocblas_double_complex* const*);

#undef INSTANTIATE_SYR_LAUNCHER_64

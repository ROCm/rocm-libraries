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

#include "handle.hpp"
#include "int64_helpers.hpp"

#include "rocblas_gbmv_64.hpp"

#include "blas2/rocblas_gbmv.hpp" // int32 API called

template <typename T, typename U, typename V>
rocblas_status rocblas_internal_gbmv_launcher_64(rocblas_handle    handle,
                                                 rocblas_operation transA,
                                                 int64_t           m_64,
                                                 int64_t           n_64,
                                                 int64_t           kl_64,
                                                 int64_t           ku_64,
                                                 const T*          alpha,
                                                 U                 A,
                                                 rocblas_stride    offsetA,
                                                 int64_t           lda_64,
                                                 rocblas_stride    strideA,
                                                 U                 x,
                                                 rocblas_stride    offsetx,
                                                 int64_t           incx_64,
                                                 rocblas_stride    stridex,
                                                 const T*          beta,
                                                 V                 y,
                                                 rocblas_stride    offsety,
                                                 int64_t           incy_64,
                                                 rocblas_stride    stridey,
                                                 int64_t           batch_count_64)
{
    // Quick return if possible. Not Argument error
    if(!m_64 || !n_64 || !batch_count_64)
        return rocblas_status_success;

    if(m_64 > c_i32_max || n_64 > c_i32_max || kl_64 > c_i32_max || ku_64 > c_i32_max)
        return rocblas_status_invalid_size; // defer adding new kernels for sizes exceeding practical memory

    for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
    {
        auto    x_ptr       = adjust_ptr_batch(x, b_base, stridex);
        auto    y_ptr       = adjust_ptr_batch(y, b_base, stridey);
        auto    A_ptr       = adjust_ptr_batch(A, b_base, strideA);
        int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

        rocblas_status status = rocblas_internal_gbmv_launcher(handle,
                                                               transA,
                                                               (int)m_64,
                                                               (int)n_64,
                                                               (int)kl_64,
                                                               (int)ku_64,
                                                               alpha,
                                                               A_ptr,
                                                               offsetA,
                                                               lda_64,
                                                               strideA,
                                                               x_ptr,
                                                               offsetx,
                                                               incx_64,
                                                               stridex,
                                                               beta,
                                                               y_ptr,
                                                               offsety,
                                                               incy_64,
                                                               stridey,
                                                               batch_count);
        if(status != rocblas_status_success)
            return status;

    } // batch

    return rocblas_status_success;
}

#define INST_GBMV_LAUNCHER(T_, U_, V_)                                     \
    template rocblas_status rocblas_internal_gbmv_launcher_64<T_, U_, V_>( \
        rocblas_handle    handle,                                          \
        rocblas_operation transA,                                          \
        int64_t           m,                                               \
        int64_t           n,                                               \
        int64_t           kl,                                              \
        int64_t           ku,                                              \
        const T_*         alpha,                                           \
        U_                A,                                               \
        rocblas_stride    offseta,                                         \
        int64_t           lda,                                             \
        rocblas_stride    strideA,                                         \
        U_                x,                                               \
        rocblas_stride    offsetx,                                         \
        int64_t           incx,                                            \
        rocblas_stride    stridex,                                         \
        const T_*         beta,                                            \
        V_                y,                                               \
        rocblas_stride    offsety,                                         \
        int64_t           incy,                                            \
        rocblas_stride    stridey,                                         \
        int64_t           batch_count);

INST_GBMV_LAUNCHER(double, double const* const*, double* const*)
INST_GBMV_LAUNCHER(rocblas_float_complex,
                   rocblas_float_complex const* const*,
                   rocblas_float_complex* const*)
INST_GBMV_LAUNCHER(rocblas_double_complex,
                   rocblas_double_complex const* const*,
                   rocblas_double_complex* const*)
INST_GBMV_LAUNCHER(float, float const*, float*)
INST_GBMV_LAUNCHER(double, double const*, double*)
INST_GBMV_LAUNCHER(rocblas_float_complex, rocblas_float_complex const*, rocblas_float_complex*)
INST_GBMV_LAUNCHER(rocblas_double_complex, rocblas_double_complex const*, rocblas_double_complex*)
INST_GBMV_LAUNCHER(float, float const* const*, float* const*)

#undef INST_GBMV_LAUNCHER

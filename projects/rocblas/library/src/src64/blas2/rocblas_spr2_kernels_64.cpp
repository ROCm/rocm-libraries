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

#include "rocblas_spr2_64.hpp"

#include "blas2/rocblas_spr2.hpp" // int32 API called

template <typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_internal_spr2_launcher_64(rocblas_handle handle,
                                                 rocblas_fill   uplo,
                                                 int64_t        n_64,
                                                 TScal const*   alpha,
                                                 TConstPtr      x,
                                                 rocblas_stride offset_x,
                                                 int64_t        incx_64,
                                                 rocblas_stride stride_x,
                                                 TConstPtr      y,
                                                 rocblas_stride offset_y,
                                                 int64_t        incy_64,
                                                 rocblas_stride stride_y,
                                                 TPtr           AP,
                                                 rocblas_stride offset_AP,
                                                 rocblas_stride stride_AP,
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
        auto    y_ptr       = adjust_ptr_batch(y, b_base, stride_y);
        auto    AP_ptr      = adjust_ptr_batch(AP, b_base, stride_AP);
        int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

        rocblas_status status = rocblas_internal_spr2_launcher(handle,
                                                               uplo,
                                                               (rocblas_int)n_64,
                                                               alpha,
                                                               x_ptr,
                                                               offset_x,
                                                               incx_64,
                                                               stride_x,
                                                               y_ptr,
                                                               offset_y,
                                                               incy_64,
                                                               stride_y,
                                                               AP_ptr,
                                                               offset_AP,
                                                               stride_AP,
                                                               batch_count);

        if(status != rocblas_status_success)
            return status;
    } // batch
    return rocblas_status_success;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files *spr2*.cpp

#ifdef INSTANTIATE_SPR2_TEMPLATE
#error INSTANTIATE_SPR2_TEMPLATE already defined
#endif

#define INSTANTIATE_SPR2_TEMPLATE(TScal_, TConstPtr_, TPtr_)                              \
    template rocblas_status rocblas_internal_spr2_launcher_64<TScal_, TConstPtr_, TPtr_>( \
        rocblas_handle handle,                                                            \
        rocblas_fill   uplo,                                                              \
        int64_t        n_64,                                                              \
        TScal_ const*  alpha,                                                             \
        TConstPtr_     x,                                                                 \
        rocblas_stride offset_x,                                                          \
        int64_t        incx_64,                                                           \
        rocblas_stride stride_x,                                                          \
        TConstPtr_     y,                                                                 \
        rocblas_stride offset_y,                                                          \
        int64_t        incy_64,                                                           \
        rocblas_stride stride_y,                                                          \
        TPtr_          AP,                                                                \
        rocblas_stride offset_AP,                                                         \
        rocblas_stride stride_AP,                                                         \
        int64_t        batch_count_64);

INSTANTIATE_SPR2_TEMPLATE(float, float const*, float*)
INSTANTIATE_SPR2_TEMPLATE(double, double const*, double*)
INSTANTIATE_SPR2_TEMPLATE(float, float const* const*, float* const*)
INSTANTIATE_SPR2_TEMPLATE(double, double const* const*, double* const*)

#undef INSTANTIATE_SPR2_TEMPLATE

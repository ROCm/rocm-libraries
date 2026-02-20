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

#include "int64_helpers.hpp"
#include "rocblas.h"
#include "rocblas_block_sizes.h"
#include "rocblas_copy_64.hpp"

#include "blas1/rocblas_copy.hpp" // int32 API called
#include "blas1/rocblas_copy_kernels.hpp"

template <typename API_INT, rocblas_int NB, typename T, typename U>
rocblas_status rocblas_internal_copy_launcher_64(rocblas_handle handle,
                                                 API_INT        n_64,
                                                 T              x,
                                                 rocblas_stride offsetx,
                                                 API_INT        incx_64,
                                                 rocblas_stride stridex,
                                                 U              y,
                                                 rocblas_stride offsety,
                                                 API_INT        incy_64,
                                                 rocblas_stride stridey,
                                                 API_INT        batch_count_64)
{
    // Quick returns handled earlier

    bool increments_32bit = std::abs(incx_64) <= c_i32_max && std::abs(incy_64) < c_i32_max;

    for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
    {
        auto    x_ptr       = adjust_ptr_batch(x, b_base, stridex);
        auto    y_ptr       = adjust_ptr_batch(y, b_base, stridey);
        int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

        for(int64_t n_base = 0; n_base < n_64; n_base += c_i64_grid_X_chunk)
        {
            int32_t n = int32_t(std::min(n_64 - n_base, c_i64_grid_X_chunk));

            int64_t shiftx
                = offsetx + (incx_64 < 0 ? -incx_64 * (n_64 - n - n_base) : n_base * incx_64);
            int64_t shifty
                = offsety + (incy_64 < 0 ? -incy_64 * (n_64 - n - n_base) : n_base * incy_64);

            if(increments_32bit)
            {
                // 32bit API call
                rocblas_status status
                    = rocblas_internal_copy_launcher<rocblas_int, NB, T, U>(handle,
                                                                            rocblas_int(n),
                                                                            x_ptr,
                                                                            shiftx,
                                                                            rocblas_int(incx_64),
                                                                            stridex,
                                                                            y_ptr,
                                                                            shifty,
                                                                            rocblas_int(incy_64),
                                                                            stridey,
                                                                            batch_count);
                if(status != rocblas_status_success)
                    return status;
            }
            else
            {
                // new instantiation of 32bit launcher for 64bit incx/y
                rocblas_status status
                    = rocblas_internal_copy_launcher<int64_t, NB, T, U>(handle,
                                                                        n,
                                                                        x_ptr,
                                                                        shiftx,
                                                                        (incx_64),
                                                                        stridex,
                                                                        y_ptr,
                                                                        shifty,
                                                                        (incy_64),
                                                                        stridey,
                                                                        batch_count);
                if(status != rocblas_status_success)
                    return status;
            }
        }
    }

    return rocblas_status_success;
}

#ifdef INSTANTIATE_COPY_LAUNCHER
#error INSTANTIATE_COPY_LAUNCHER already defined
#endif

#define INSTANTIATE_COPY_LAUNCHER(NB_, T_, U_)                                       \
    template rocblas_status rocblas_internal_copy_launcher_64<int64_t, NB_, T_, U_>( \
        rocblas_handle handle,                                                       \
        int64_t        n,                                                            \
        T_             x,                                                            \
        rocblas_stride offsetx,                                                      \
        int64_t        incx,                                                         \
        rocblas_stride stridex,                                                      \
        U_             y,                                                            \
        rocblas_stride offsety,                                                      \
        int64_t        incy,                                                         \
        rocblas_stride stridey,                                                      \
        int64_t        batch_count);

// non batched

INSTANTIATE_COPY_LAUNCHER(ROCBLAS_COPY_NB, const float*, float*)
INSTANTIATE_COPY_LAUNCHER(ROCBLAS_COPY_NB, const double*, double*)
INSTANTIATE_COPY_LAUNCHER(ROCBLAS_COPY_NB, const rocblas_half*, rocblas_half*)
INSTANTIATE_COPY_LAUNCHER(ROCBLAS_COPY_NB, const rocblas_float_complex*, rocblas_float_complex*)
INSTANTIATE_COPY_LAUNCHER(ROCBLAS_COPY_NB, const rocblas_double_complex*, rocblas_double_complex*)

// batched

INSTANTIATE_COPY_LAUNCHER(ROCBLAS_COPY_NB, float const* const*, float* const*)
INSTANTIATE_COPY_LAUNCHER(ROCBLAS_COPY_NB, double const* const*, double* const*)
INSTANTIATE_COPY_LAUNCHER(ROCBLAS_COPY_NB, rocblas_half const* const*, rocblas_half* const*)
INSTANTIATE_COPY_LAUNCHER(ROCBLAS_COPY_NB,
                          rocblas_float_complex const* const*,
                          rocblas_float_complex* const*)
INSTANTIATE_COPY_LAUNCHER(ROCBLAS_COPY_NB,
                          rocblas_double_complex const* const*,
                          rocblas_double_complex* const*)

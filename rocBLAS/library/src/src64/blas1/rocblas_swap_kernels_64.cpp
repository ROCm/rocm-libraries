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
 * SWAPRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#include "handle.hpp"
#include "int64_helpers.hpp"
#include "rocblas_block_sizes.h"
#include "rocblas_swap_64.hpp"

#include "blas1/rocblas_swap.hpp" // rocblas_int API called
#include "blas1/rocblas_swap_kernels.hpp" // inst kernels with int64_t

template <typename API_INT, int NB, typename T>
rocblas_status rocblas_internal_swap_launcher_64(rocblas_handle handle,
                                                 API_INT        n_64,
                                                 T              x,
                                                 rocblas_stride offsetx,
                                                 API_INT        incx_64,
                                                 rocblas_stride stridex,
                                                 T              y,
                                                 rocblas_stride offsety,
                                                 API_INT        incy_64,
                                                 rocblas_stride stridey,
                                                 API_INT        batch_count_64)
{
    if(std::abs(incx_64) <= c_i32_max && std::abs(incy_64) < c_i32_max)
    {
        if(n_64 <= c_ILP64_i32_max && batch_count_64 < c_i64_grid_YZ_chunk)
        {
            // valid to use original 32bit API with truncated 64bit args
            return rocblas_internal_swap_launcher<rocblas_int, NB, T>(handle,
                                                                      n_64,
                                                                      x,
                                                                      offsetx,
                                                                      incx_64,
                                                                      stridex,
                                                                      y,
                                                                      offsety,
                                                                      incy_64,
                                                                      stridey,
                                                                      batch_count_64);
        }

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

                // 32bit API call
                rocblas_status status
                    = rocblas_internal_swap_launcher<rocblas_int, NB, T>(handle,
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
        }
    }
    else
    {
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

                // new instantiation for 64bit incx/y
                rocblas_status status = rocblas_internal_swap_launcher<int64_t, NB, T>(handle,
                                                                                       n,
                                                                                       x_ptr,
                                                                                       shiftx,
                                                                                       incx_64,
                                                                                       stridex,
                                                                                       y_ptr,
                                                                                       shifty,
                                                                                       incy_64,
                                                                                       stridey,
                                                                                       batch_count);
                if(status != rocblas_status_success)
                    return status;
            }
        }
    }
    return rocblas_status_success;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files swap*.cpp

// clang-format off
#ifdef INSTANTIATE_SWAP_LAUNCHER
#error INSTANTIATE_SWAP_LAUNCHER already defined
#endif

#define INSTANTIATE_SWAP_LAUNCHER(NB_, T_)                                           \
    template rocblas_status rocblas_internal_swap_launcher_64<int64_t, NB_, T_>(     \
        rocblas_handle handle,                                                       \
        int64_t        n,                                                            \
        T_             x,                                                            \
        rocblas_stride offsetx,                                                      \
        int64_t        incx,                                                         \
        rocblas_stride stridex,                                                      \
        T_             y,                                                            \
        rocblas_stride offsety,                                                      \
        int64_t        incy,                                                         \
        rocblas_stride stridey,                                                      \
        int64_t        batch_count);

// non batched

INSTANTIATE_SWAP_LAUNCHER(ROCBLAS_SWAP_NB, float*)
INSTANTIATE_SWAP_LAUNCHER(ROCBLAS_SWAP_NB, double*)
INSTANTIATE_SWAP_LAUNCHER(ROCBLAS_SWAP_NB, rocblas_float_complex*)
INSTANTIATE_SWAP_LAUNCHER(ROCBLAS_SWAP_NB, rocblas_double_complex*)

// batched

INSTANTIATE_SWAP_LAUNCHER(ROCBLAS_SWAP_NB, float* const*)
INSTANTIATE_SWAP_LAUNCHER(ROCBLAS_SWAP_NB, double* const*)
INSTANTIATE_SWAP_LAUNCHER(ROCBLAS_SWAP_NB,
                          rocblas_float_complex* const*)
INSTANTIATE_SWAP_LAUNCHER(ROCBLAS_SWAP_NB,
                          rocblas_double_complex* const*)

// clang-format on

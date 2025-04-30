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
#include "rocblas_rotm_64.hpp"

#include "blas1/rocblas_rotm.hpp" // int32 API called

template <typename API_INT, int NB, bool BATCHED_OR_STRIDED, typename T, typename U>
ROCBLAS_INTERNAL_ONLY_EXPORT_NOINLINE rocblas_status
    rocblas_internal_rotm_launcher(rocblas_handle handle,
                                   API_INT        n,
                                   T              x,
                                   rocblas_stride offset_x,
                                   int64_t        incx,
                                   rocblas_stride stride_x,
                                   T              y,
                                   rocblas_stride offset_y,
                                   int64_t        incy,
                                   rocblas_stride stride_y,
                                   U              param,
                                   rocblas_stride offset_param,
                                   rocblas_stride stride_param,
                                   API_INT        batch_count);

template <typename API_INT, int NB, bool BATCHED_OR_STRIDED, typename T, typename U>
rocblas_status rocblas_internal_rotm_launcher(rocblas_handle handle,
                                              API_INT        n_64,
                                              T              x,
                                              rocblas_stride offsetx,
                                              int64_t        incx_64,
                                              rocblas_stride stridex,
                                              T              y,
                                              rocblas_stride offsety,
                                              int64_t        incy_64,
                                              rocblas_stride stridey,
                                              U              param,
                                              rocblas_stride offset_param,
                                              rocblas_stride stride_param,
                                              API_INT        batch_count_64)
{
    // Quick returns handled earlier

    for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
    {
        auto x_ptr = adjust_ptr_batch(x, b_base, stridex);
        auto y_ptr = adjust_ptr_batch(y, b_base, stridey);
        auto p_ptr = adjust_ptr_batch(param, b_base, stride_param);

        int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

        for(int64_t n_base = 0; n_base < n_64; n_base += c_i64_grid_X_chunk)
        {
            int32_t n = int32_t(std::min(n_64 - n_base, c_i64_grid_X_chunk));

            int64_t shiftx
                = offsetx + (incx_64 < 0 ? -incx_64 * (n_64 - n - n_base) : n_base * incx_64);
            int64_t shifty
                = offsety + (incy_64 < 0 ? -incy_64 * (n_64 - n - n_base) : n_base * incy_64);

            // 32bit API call as incx/y int64_t
            rocblas_status status
                = rocblas_internal_rotm_launcher<rocblas_int, NB, BATCHED_OR_STRIDED, T, U>(
                    handle,
                    rocblas_int(n),
                    x_ptr,
                    shiftx,
                    incx_64,
                    stridex,
                    y_ptr,
                    shifty,
                    incy_64,
                    stridey,
                    p_ptr,
                    offset_param,
                    stride_param,
                    batch_count);
            if(status != rocblas_status_success)
                return status;
        }
    }

    return rocblas_status_success;
}

#ifdef INST_ROTM_LAUNCHER
#error INST_ROTM_LAUNCHER already defined
#endif

#define INST_ROTM_LAUNCHER(TI_, NB_, BATCHED_OR_STRIDED_, T_, U_)                                  \
    template rocblas_status rocblas_internal_rotm_launcher<TI_, NB_, BATCHED_OR_STRIDED_, T_, U_>( \
        rocblas_handle handle,                                                                     \
        TI_            n,                                                                          \
        T_             x,                                                                          \
        rocblas_stride offsetx,                                                                    \
        int64_t        incx,                                                                       \
        rocblas_stride stridex,                                                                    \
        T_             y,                                                                          \
        rocblas_stride offsety,                                                                    \
        int64_t        incy,                                                                       \
        rocblas_stride stridey,                                                                    \
        U_             param,                                                                      \
        rocblas_stride offset_param,                                                               \
        rocblas_stride stride_param,                                                               \
        TI_            batch_count);

// instantiate for rocblas_Xrotm and rocblas_Xrotm_strided_batched
INST_ROTM_LAUNCHER(int64_t, ROCBLAS_ROTM_NB, true, float*, float const*);
INST_ROTM_LAUNCHER(int64_t, ROCBLAS_ROTM_NB, false, float*, float const*);
INST_ROTM_LAUNCHER(int64_t, ROCBLAS_ROTM_NB, true, double*, double const*);
INST_ROTM_LAUNCHER(int64_t, ROCBLAS_ROTM_NB, false, double*, double const*);

// instantiate for rocblas_Xrotm__batched
INST_ROTM_LAUNCHER(int64_t, ROCBLAS_ROTM_NB, true, float* const*, float const* const*);
INST_ROTM_LAUNCHER(int64_t, ROCBLAS_ROTM_NB, true, double* const*, double const* const*);

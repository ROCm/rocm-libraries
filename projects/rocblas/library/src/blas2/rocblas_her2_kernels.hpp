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

#include "check_numerics_matrix.hpp"
#include "check_numerics_vector.hpp"
#include "device_macros.hpp"
#include "handle.hpp"
#include "rocblas_her2.hpp"

template <typename API_INT, typename T>
__forceinline__ __device__ void rocblas_her2_kernel_calc(bool        is_upper,
                                                         rocblas_int n,
                                                         size_t      area,
                                                         T           alpha,
                                                         const T*    x,
                                                         API_INT     incx,
                                                         const T*    y,
                                                         API_INT     incy,
                                                         T*          A,
                                                         API_INT     lda)
{
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; // linear area index
    if(i >= area)
        return;

    size_t ri = !is_upper ? area - 1 - i : i;

    // linearized triangle with diagonal to col, row
    int k  = int(sqrt(8 * ri + 1) - 1) / 2;
    int ty = k;
    int tx = ri - k * size_t(k + 1) / 2;

    if(!is_upper)
    {
        int maxIdx = n - 1;
        tx         = maxIdx - tx;
        ty         = maxIdx - ty;
    }

    if(is_upper ? tx < ty : ty < tx)
    {
        A[tx + int64_t(lda) * ty]
            += alpha * x[tx * int64_t(incx)] * conj(y[ty * int64_t(incy)])
               + conj(alpha) * y[tx * int64_t(incy)] * conj(x[ty * int64_t(incx)]);
    }
    else if(tx == ty)
    {
        A[tx + int64_t(lda) * ty]
            = std::real(A[tx + int64_t(lda) * ty])
              + alpha * x[tx * int64_t(incx)] * conj(y[ty * int64_t(incy)])
              + conj(alpha) * y[tx * int64_t(incy)] * conj(x[ty * int64_t(incx)]);
    }
}

template <typename API_INT, int DIM_X, typename TScal, typename TConstPtr, typename TPtr>
ROCBLAS_KERNEL(DIM_X)
rocblas_her2_kernel(bool           is_upper,
                    rocblas_int    n,
                    size_t         area,
                    TScal          alpha_device_host,
                    TConstPtr      xa,
                    rocblas_stride shift_x,
                    API_INT        incx,
                    rocblas_stride stride_x,
                    TConstPtr      ya,
                    rocblas_stride shift_y,
                    API_INT        incy,
                    rocblas_stride stride_y,
                    TPtr           Aa,
                    rocblas_stride shift_A,
                    API_INT        lda,
                    rocblas_stride stride_A,
                    rocblas_int    batch_count)
{

    auto alpha = load_scalar(alpha_device_host);

    if(!alpha)
        return;

    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif
        auto*       A = load_ptr_batch(Aa, batch, shift_A, stride_A);
        const auto* x = load_ptr_batch(xa, batch, shift_x, stride_x);
        const auto* y = load_ptr_batch(ya, batch, shift_y, stride_y);

        rocblas_her2_kernel_calc(is_upper, n, area, alpha, x, incx, y, incy, A, lda);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

/**
 * TScal     is always: const T* (either2 host or device)
 * TConstPtr is either2: const T* OR const T* const*
 * TPtr      is either2:       T* OR       T* const*
 * Where T is the base type (rocblas_float_complex or rocblas_double_complex)
 */
template <typename API_INT, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_her2_launcher(rocblas_handle handle,
                                     rocblas_fill   uplo,
                                     rocblas_int    n,
                                     TScal          alpha,
                                     TConstPtr      x,
                                     rocblas_stride offset_x,
                                     API_INT        incx,
                                     rocblas_stride stride_x,
                                     TConstPtr      y,
                                     rocblas_stride offset_y,
                                     API_INT        incy,
                                     rocblas_stride stride_y,
                                     TPtr           A,
                                     rocblas_stride offset_A,
                                     API_INT        lda,
                                     rocblas_stride stride_A,
                                     rocblas_int    batch_count)
{
    // Quick return if possible. Not Argument error
    if(!n || !batch_count)
        return rocblas_status_success;

    // in case of negative inc, shift pointer to end of data for negative indexing tid*inc
    int64_t shift_x = incx < 0 ? offset_x - int64_t(incx) * (n - 1) : offset_x;
    int64_t shift_y = incy < 0 ? offset_y - int64_t(incy) * (n - 1) : offset_y;

    int batches = handle->getBatchGridDim((int)batch_count);

    static constexpr int HER2_DIM_X = 512;

    size_t nitems = (size_t)n * (n + 1) / 2;

    rocblas_int blocksX = (nitems - 1) / (HER2_DIM_X) + 1;

    dim3 her2_grid(blocksX, 1, batches);
    dim3 her2_threads(HER2_DIM_X);

    if(rocblas_pointer_mode_device == handle->pointer_mode)
    {
        ROCBLAS_LAUNCH_KERNEL((rocblas_her2_kernel<API_INT, HER2_DIM_X>),
                              her2_grid,
                              her2_threads,
                              0,
                              handle->get_stream(),
                              uplo == rocblas_fill_upper,
                              n,
                              nitems,
                              alpha,
                              x,
                              shift_x,
                              incx,
                              stride_x,
                              y,
                              shift_y,
                              incy,
                              stride_y,
                              A,
                              offset_A,
                              lda,
                              stride_A,
                              batch_count);
    }
    else
        ROCBLAS_LAUNCH_KERNEL((rocblas_her2_kernel<API_INT, HER2_DIM_X>),
                              her2_grid,
                              her2_threads,
                              0,
                              handle->get_stream(),
                              uplo == rocblas_fill_upper,
                              n,
                              nitems,
                              *alpha,
                              x,
                              shift_x,
                              incx,
                              stride_x,
                              y,
                              shift_y,
                              incy,
                              stride_y,
                              A,
                              offset_A,
                              lda,
                              stride_A,
                              batch_count);

    return rocblas_status_success;
}

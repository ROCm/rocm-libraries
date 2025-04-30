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

#include "device_macros.hpp"
#include "handle.hpp"
#include "int64_helpers.hpp"
#include "rocblas.h"
#include "rocblas_rot.hpp"

template <typename API_INT,
          rocblas_int NB,
          typename Tex,
          typename Tx,
          typename Ty,
          typename Tc,
          typename Ts>
rocblas_status rocblas_internal_rot_launcher(rocblas_handle handle,
                                             API_INT        n,
                                             Tx             x,
                                             rocblas_stride offset_x,
                                             int64_t        incx,
                                             rocblas_stride stride_x,
                                             Ty             y,
                                             rocblas_stride offset_y,
                                             int64_t        incy,
                                             rocblas_stride stride_y,
                                             Tc*            c,
                                             rocblas_stride c_stride,
                                             Ts*            s,
                                             rocblas_stride s_stride,
                                             API_INT        batch_count);

template <typename API_INT,
          typename Tex,
          typename Tx,
          typename Ty,
          typename Tc,
          typename Ts,
          std::enable_if_t<!rocblas_is_complex<Ts>, int> = 0>
__forceinline__ __device__ void
    rocblas_rot_kernel_calc(rocblas_int n, Tx* x, int64_t incx, Ty* y, int64_t incy, Tc c, Ts s)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < n)
    {
        int64_t ix    = tid * incx;
        int64_t iy    = tid * incy;
        Tex     tempx = Tex(c * x[ix]) + Tex(s * y[iy]);
        Tex     tempy = Tex(c * y[iy]) - Tex((s)*x[ix]);
        y[iy]         = Ty(tempy);
        x[ix]         = Tx(tempx);
    }
}

template <typename API_INT,
          typename Tex,
          typename Tx,
          typename Ty,
          typename Tc,
          typename Ts,
          std::enable_if_t<rocblas_is_complex<Ts>, int> = 0>
__forceinline__ __device__ void
    rocblas_rot_kernel_calc(rocblas_int n, Tx* x, int64_t incx, Ty* y, int64_t incy, Tc c, Ts s)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < n)
    {
        int64_t ix    = tid * incx;
        int64_t iy    = tid * incy;
        Tex     tempx = Tex(c * x[ix]) + Tex(s * y[iy]);
        Tex     tempy = Tex(c * y[iy]) - Tex(conj(s) * x[ix]);
        y[iy]         = Ty(tempy);
        x[ix]         = Tx(tempx);
    }
}

template <typename API_INT,
          rocblas_int NB,
          typename Tex,
          typename Tx,
          typename Ty,
          typename Tc,
          typename Ts>
ROCBLAS_KERNEL(NB)
rocblas_rot_kernel(rocblas_int    n,
                   Tx             x_in,
                   rocblas_stride offset_x,
                   int64_t        incx,
                   rocblas_stride stride_x,
                   Ty             y_in,
                   rocblas_stride offset_y,
                   int64_t        incy,
                   rocblas_stride stride_y,
                   Tc             c_in,
                   rocblas_stride c_stride,
                   Ts             s_in,
                   rocblas_stride s_stride,
                   rocblas_int    batch_count)
{
    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        auto c = std::real(load_scalar(c_in, batch, c_stride));
        auto s = load_scalar(s_in, batch, s_stride);
        auto x = load_ptr_batch(x_in, batch, offset_x, stride_x);
        auto y = load_ptr_batch(y_in, batch, offset_y, stride_y);

        rocblas_rot_kernel_calc<API_INT, Tex>(n, x, incx, y, incy, c, s);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

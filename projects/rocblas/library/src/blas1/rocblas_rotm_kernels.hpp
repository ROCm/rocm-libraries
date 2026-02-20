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

#include "check_numerics_vector.hpp"
#include "device_macros.hpp"
#include "handle.hpp"
#include "logging.hpp"
#include "rocblas_block_sizes.h"
#include "rocblas_rotm.hpp"

template <typename T, typename U>
__forceinline__ __device__ void rocblas_rotm_kernel_calc(rocblas_int    n,
                                                         T              x_in,
                                                         rocblas_stride offset_x,
                                                         int64_t        incx,
                                                         rocblas_stride stride_x,
                                                         T              y_in,
                                                         rocblas_stride offset_y,
                                                         int64_t        incy,
                                                         rocblas_stride stride_y,
                                                         U              flag,
                                                         U              h11,
                                                         U              h21,
                                                         U              h12,
                                                         U              h22,
                                                         rocblas_int    batch)
{
    auto    x   = load_ptr_batch(x_in, batch, offset_x, stride_x);
    auto    y   = load_ptr_batch(y_in, batch, offset_y, stride_y);
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < n && flag != -2)
    {
        int64_t ix = tid * incx;
        int64_t iy = tid * incy;
        auto    w  = x[ix];
        auto    z  = y[iy];
        if(flag < 0)
        {
            //cppcheck-suppress unreadVariable # The variables 'x' and 'y' will be copied back to host(CPU)
            x[ix] = w * h11 + z * h12;
            //cppcheck-suppress unreadVariable # The variables 'x' and 'y' will be copied back to host(CPU)
            y[iy] = w * h21 + z * h22;
        }
        else if(flag == 0)
        {
            //cppcheck-suppress unreadVariable # The variables 'x' and 'y' will be copied back to host(CPU)
            x[ix] = w + z * h12;
            //cppcheck-suppress unreadVariable # The variables 'x' and 'y' will be copied back to host(CPU)
            y[iy] = w * h21 + z;
        }
        else
        {
            //cppcheck-suppress unreadVariable # The variables 'x' and 'y' will be copied back to host(CPU)
            x[ix] = w * h11 + z;
            //cppcheck-suppress unreadVariable # The variables 'x' and 'y' will be copied back to host(CPU)
            y[iy] = -w + z * h22;
        }
    }
}

template <int NB, typename T, typename U>
ROCBLAS_KERNEL(NB)
rocblas_rotm_kernel_batched(rocblas_int    n,
                            T              x_in,
                            rocblas_stride offset_x,
                            int64_t        incx,
                            rocblas_stride stride_x,
                            T              y_in,
                            rocblas_stride offset_y,
                            int64_t        incy,
                            rocblas_stride stride_y,
                            U              param,
                            rocblas_stride offset_param,
                            rocblas_stride stride_param,
                            rocblas_int    batch_count)
{
    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        auto p    = load_ptr_batch(param, batch, offset_param, stride_param);
        auto flag = p[0];
        auto h11  = p[1];
        auto h21  = p[2];
        auto h12  = p[3];
        auto h22  = p[4];
        rocblas_rotm_kernel_calc(n,
                                 x_in,
                                 offset_x,
                                 incx,
                                 stride_x,
                                 y_in,
                                 offset_y,
                                 incy,
                                 stride_y,
                                 flag,
                                 h11,
                                 h21,
                                 h12,
                                 h22,
                                 batch);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <int NB, typename T, typename U>
ROCBLAS_KERNEL(NB)
rocblas_rotm_kernel_regular(rocblas_int    n,
                            T*             x_in,
                            rocblas_stride offset_x,
                            int64_t        incx,
                            rocblas_stride stride_x,
                            T*             y_in,
                            rocblas_stride offset_y,
                            int64_t        incy,
                            rocblas_stride stride_y,
                            U              flag,
                            U              h11,
                            U              h21,
                            U              h12,
                            U              h22)
{
    rocblas_rotm_kernel_calc(n,
                             x_in,
                             offset_x,
                             incx,
                             stride_x,
                             y_in,
                             offset_y,
                             incy,
                             stride_y,
                             load_scalar(flag),
                             load_scalar(h11),
                             load_scalar(h21),
                             load_scalar(h12),
                             load_scalar(h22),
                             0);
}

// Workaround to avoid constexpr if - Helper function to quick return when param[0] == -2
template <typename T>
inline bool rocblas_rotm_quick_return_param(rocblas_handle handle,
                                            const T*       param,
                                            rocblas_stride stride_param)
{
    if(rocblas_pointer_mode_host == handle->pointer_mode)
        if(param[0] == -2 && stride_param == 0)
            return true;
    return false;
}

template <typename T>
inline bool rocblas_rotm_quick_return_param(rocblas_handle handle,
                                            const T* const param[],
                                            rocblas_stride stride_param)
{
    return false;
}

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

template <typename API_INT, int DIM_X, typename T, typename Tex, typename Ta, typename Tx>
ROCBLAS_KERNEL(DIM_X)
rocblas_scal_kernel(rocblas_int    n,
                    Ta             alpha_device_host,
                    rocblas_stride stride_alpha,
                    Tx             xa,
                    rocblas_stride offset_x,
                    API_INT        incx,
                    rocblas_stride stride_x,
                    rocblas_int    batch_count)
{
    uint32_t tid   = blockIdx.x * DIM_X + threadIdx.x;
    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        auto* x     = load_ptr_batch(xa, batch, offset_x, stride_x);
        auto  alpha = load_scalar(alpha_device_host, batch, stride_alpha);

        if(alpha != 1 && tid < n)
        {

            Tex res                = (Tex)x[tid * int64_t(incx)] * alpha;
            x[tid * int64_t(incx)] = (T)res;
        }

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

//!
//! @brief Optimized kernel for the SCAL floating points.
//! @remark Increment are required to be equal to one, that's why they are unspecified.
//!
template <int DIM_X, typename T, typename Tex, typename Ta, typename Tx>
ROCBLAS_KERNEL(DIM_X)
rocblas_sscal_2_kernel(rocblas_int    n,
                       Ta             alpha_device_host,
                       rocblas_stride stride_alpha,
                       Tx __restrict__ xa,
                       rocblas_stride offset_x,
                       rocblas_stride stride_x,
                       rocblas_int    batch_count)
{
    uint32_t tid   = (blockIdx.x * DIM_X + threadIdx.x) * 2;
    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        auto* x     = load_ptr_batch(xa, batch, offset_x, stride_x);
        auto  alpha = load_scalar(alpha_device_host, batch, stride_alpha);

        if(alpha != 1)
        {
            if(tid + 1 < n)
            {
                // Each thread access contiguous elements for example Thread '0' access indices '0' and '1' of the vector `x`
                for(int32_t j = 0; j < 2; ++j)
                {
                    Tex res    = (Tex)x[tid + j] * alpha;
                    x[tid + j] = (T)res;
                }
            }

            // If `n` is odd then the computation of last element in the vector `x` is covered below.
            if(n % 2 != 0 && tid == n - 1)
            {
                Tex res = (Tex)x[tid] * alpha;
                x[tid]  = (T)res;
            }
        }

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

//!
//! @brief Optimized kernel for the SCAL when the compute and alpha type is half precision.
//! @remark Increments are required to be equal to one, that's why they are unspecified.
//!
template <int DIM_X, typename Ta, typename Tx>
ROCBLAS_KERNEL(DIM_X)
rocblas_hscal_mlt_4_kernel(rocblas_int    n,
                           rocblas_int    n_mod_4,
                           rocblas_int    n_mlt_4,
                           Ta             alpha_device_host,
                           rocblas_stride stride_alpha,
                           Tx __restrict__ xa,
                           rocblas_stride offset_x,
                           rocblas_stride stride_x,
                           rocblas_int    batch_count)
{
    uint32_t tid   = (blockIdx.x * DIM_X + threadIdx.x) * 4;
    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        auto alpha = load_scalar(alpha_device_host, batch, stride_alpha);

        if(alpha != 1)
        {
            rocblas_half2 x0, x1;
            rocblas_half2 z0, z1;

            if(tid + 3 < n)
            {
                rocblas_half4* x
                    = (rocblas_half4*)load_ptr_batch(xa, batch, offset_x + tid, stride_x);

                x0[0] = (*x)[0];
                x0[1] = (*x)[1];
                x1[0] = (*x)[2];
                x1[1] = (*x)[3];

                z0[0] = alpha * x0[0];
                z0[1] = alpha * x0[1];
                z1[0] = alpha * x1[0];
                z1[1] = alpha * x1[1];

                (*x)[0] = z0[0];
                (*x)[1] = z0[1];
                (*x)[2] = z1[0];
                (*x)[3] = z1[1];
            }

            // If `n_mod_4` is true then the computation of last few element in the vector `x` is covered below.
            if(n_mod_4)
            {
                //The last ThreadID which is a multiple of 4 should complete the computation of last few elements of vector `x`
                if(tid == n_mlt_4)
                {
                    auto* x = load_ptr_batch(xa, batch, offset_x, stride_x);
                    for(int32_t j = 0; j < n_mod_4; ++j)
                    {
                        x[tid + j] = x[tid + j] * alpha;
                    }
                }
            }
        }

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <typename API_INT, int NB, typename T, typename Tex, typename Ta, typename Tx>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_scal_launcher(rocblas_handle handle,
                                   API_INT        n,
                                   const Ta*      alpha,
                                   rocblas_stride stride_alpha,
                                   Tx             x,
                                   rocblas_stride offset_x,
                                   API_INT        incx,
                                   rocblas_stride stride_x,
                                   API_INT        batch_count)
{
    // Quick return if possible. Not Argument error
    if(n <= 0 || incx <= 0 || batch_count <= 0)
    {
        return rocblas_status_success;
    }

    static constexpr bool using_rocblas_float
        = std::is_same_v<Tx, rocblas_float*> || std::is_same_v<Tx, rocblas_float* const*>;

    // Using rocblas_half ?
    static constexpr bool using_rocblas_half
        = std::is_same_v<Ta, rocblas_half> && std::is_same_v<Tex, rocblas_half>;

    int batches = handle->getBatchGridDim((int)batch_count);

    if(using_rocblas_float && incx == 1)
    {
        // Kernel function for improving the performance of SSCAL when incx==1
        int32_t blocks = (n - 1) / (NB * 2) + 1;
        dim3    grid(blocks, 1, batches);
        dim3    threads(NB);

        if(rocblas_pointer_mode_device == handle->pointer_mode)
            ROCBLAS_LAUNCH_KERNEL((rocblas_sscal_2_kernel<NB, T, Tex>),
                                  grid,
                                  threads,
                                  0,
                                  handle->get_stream(),
                                  n,
                                  alpha,
                                  stride_alpha,
                                  x,
                                  offset_x,
                                  stride_x,
                                  batch_count);
        else // single alpha is on host
            ROCBLAS_LAUNCH_KERNEL((rocblas_sscal_2_kernel<NB, T, Tex>),
                                  grid,
                                  threads,
                                  0,
                                  handle->get_stream(),
                                  n,
                                  *alpha,
                                  stride_alpha,
                                  x,
                                  offset_x,
                                  stride_x,
                                  batch_count);
    }
    else if(using_rocblas_half && incx == 1)
    {
        // Kernel function for improving the performance of HSCAL when incx==1
        int32_t n_mod_4 = n & 3; // n mod 4
        int32_t n_mlt_4 = n & ~(rocblas_int)3; // multiple of 4
        int32_t blocks  = (n - 1) / (NB * 4) + 1;
        dim3    grid(blocks, 1, batches);
        dim3    threads(NB);

        if constexpr(using_rocblas_half)
        {
            if(rocblas_pointer_mode_device == handle->pointer_mode)
                ROCBLAS_LAUNCH_KERNEL((rocblas_hscal_mlt_4_kernel<NB>),
                                      grid,
                                      threads,
                                      0,
                                      handle->get_stream(),
                                      n,
                                      n_mod_4,
                                      n_mlt_4,
                                      (const rocblas_half*)alpha,
                                      stride_alpha,
                                      x,
                                      offset_x,
                                      stride_x,
                                      batch_count);
            else // single alpha is on host
                ROCBLAS_LAUNCH_KERNEL((rocblas_hscal_mlt_4_kernel<NB>),
                                      grid,
                                      threads,
                                      0,
                                      handle->get_stream(),
                                      n,
                                      n_mod_4,
                                      n_mlt_4,
                                      load_scalar((const rocblas_half*)alpha),
                                      stride_alpha,
                                      x,
                                      offset_x,
                                      stride_x,
                                      batch_count);
        }
    }
    else
    {
        int  blocks = (n - 1) / NB + 1;
        dim3 grid(blocks, 1, batches);
        dim3 threads(NB);

        if(rocblas_pointer_mode_device == handle->pointer_mode)
            ROCBLAS_LAUNCH_KERNEL((rocblas_scal_kernel<API_INT, NB, T, Tex>),
                                  grid,
                                  threads,
                                  0,
                                  handle->get_stream(),
                                  n,
                                  alpha,
                                  stride_alpha,
                                  x,
                                  offset_x,
                                  incx,
                                  stride_x,
                                  batch_count);
        else // single alpha is on host
            ROCBLAS_LAUNCH_KERNEL((rocblas_scal_kernel<API_INT, NB, T, Tex>),
                                  grid,
                                  threads,
                                  0,
                                  handle->get_stream(),
                                  n,
                                  *alpha,
                                  stride_alpha,
                                  x,
                                  offset_x,
                                  incx,
                                  stride_x,
                                  batch_count);
    }
    return rocblas_status_success;
}

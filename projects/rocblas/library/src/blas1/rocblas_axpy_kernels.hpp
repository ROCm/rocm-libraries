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
#include "rocblas_axpy.hpp"

//!
//! @brief General kernel (batched, strided batched) of axpy.
//!
template <typename API_INT, rocblas_int NB, typename Tex, typename Ta, typename Tx, typename Ty>
ROCBLAS_KERNEL(NB)
rocblas_axpy_kernel(rocblas_int    n,
                    Ta             alpha_device_host,
                    rocblas_stride stride_alpha,
                    Tx __restrict__ x,
                    rocblas_stride offset_x,
                    API_INT        incx,
                    rocblas_stride stride_x,
                    Ty __restrict__ y,
                    rocblas_stride offset_y,
                    API_INT        incy,
                    rocblas_stride stride_y,
                    rocblas_int    batch_count)
{
    int64_t  tid   = blockIdx.x * NB + threadIdx.x;
    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif
        auto alpha = load_scalar(alpha_device_host, batch, stride_alpha);
        if(alpha)
        {
            if(tid < n)
            {
                auto tx = load_ptr_batch(x, batch, offset_x + tid * incx, stride_x);
                auto ty = load_ptr_batch(y, batch, offset_y + tid * incy, stride_y);

                *ty = (*ty) + Tex(alpha) * (*tx);
            }
        }

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

//!
//! @brief Optimized kernel for the AXPY floating points.
//! @remark Increment are required to be equal to one, that's why they are unspecified.
//!
template <rocblas_int NB, typename Tex, typename Ta, typename Tx, typename Ty>
ROCBLAS_KERNEL(NB)
rocblas_saxpy_2_kernel(rocblas_int    n,
                       Ta             alpha_device_host,
                       rocblas_stride stride_alpha,
                       Tx __restrict__ x,
                       rocblas_stride offset_x,
                       rocblas_stride stride_x,
                       Ty __restrict__ y,
                       rocblas_stride offset_y,
                       rocblas_stride stride_y,
                       rocblas_int    batch_count)
{

    int64_t  tid   = (blockIdx.x * NB + threadIdx.x) * 2;
    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        auto alpha = load_scalar(alpha_device_host, batch, stride_alpha);
        if(alpha)
        {
            auto* tx = load_ptr_batch(x, batch, offset_x, stride_x);
            auto* ty = load_ptr_batch(y, batch, offset_y, stride_y);

            if(tid < n - 1)
            {
                // Each thread access contiguous elements for example Thread '0' access indices '0' and '1' of the vectors `x` and `y`
                for(int j = 0; j < 2; ++j)
                {
                    ty[tid + j] = ty[tid + j] + Tex(alpha) * tx[tid + j];
                }
            }

            // If `n` is odd then the computation of last element in the vectors is covered below.
            if(n % 2 != 0 && tid == n - 1)
            {
                ty[tid] = ty[tid] + Tex(alpha) * tx[tid];
            }
        }

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

//!
//! @brief Large batch size kernel (batched, strided batched) of axpy.
//!
template <typename API_INT,
          int DIM_X,
          int DIM_Y,
          typename Tex,
          typename Ta,
          typename Tx,
          typename Ty>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_axpy_kernel_batched(rocblas_int    n,
                            Ta             alpha_device_host,
                            rocblas_stride stride_alpha,
                            Tx             x,
                            rocblas_stride offset_x,
                            API_INT        incx,
                            rocblas_stride stride_x,
                            Ty             y,
                            rocblas_stride offset_y,
                            API_INT        incy,
                            rocblas_stride stride_y,
                            rocblas_int    batch_count)
{
    int64_t  tid   = blockIdx.x * DIM_X + threadIdx.x;
    uint32_t batch = (blockIdx.z * DIM_Y + threadIdx.y) * 4;

    if(tid < n)
    {
        int64_t ix = tid * incx;
        int64_t iy = tid * incy;

#if DEVICE_GRID_YZ_16BIT
        for(; batch < batch_count; batch += gridDim.z * DIM_Y * 4)
        {
#endif

            for(int i = 0; i < 4; i++)
            {
                if(batch + i < batch_count)
                {
                    auto alpha = load_scalar(alpha_device_host, batch + i, stride_alpha);
                    if(alpha)
                    {
                        Tex ex_alph = Tex(alpha);

                        auto tx = load_ptr_batch(x, batch + i, offset_x + ix, stride_x);
                        auto ty = load_ptr_batch(y, batch + i, offset_y + iy, stride_y);

                        *ty = (*ty) + ex_alph * (*tx);
                    }
                }
            }

#if DEVICE_GRID_YZ_16BIT
        }
#endif
    }
}

//!
//! @brief Optimized kernel for the remaining part of 8 half floating points.
//! @remark Increment are required to be equal to one, that's why they are unspecified.
//!
template <rocblas_int NB, typename Ta, typename Tx, typename Ty>
ROCBLAS_KERNEL(NB)
rocblas_haxpy_mod_8_kernel(rocblas_int    n_mod_8,
                           Ta             alpha_device_host,
                           rocblas_stride stride_alpha,
                           Tx             x,
                           int64_t        offset_x,
                           rocblas_stride stride_x,
                           Ty             y,
                           int64_t        offset_y,
                           rocblas_stride stride_y,
                           rocblas_int    batch_count)
{
    int64_t  tid   = blockIdx.x * NB + threadIdx.x;
    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        auto alpha = load_scalar(alpha_device_host, batch, stride_alpha);
        if(alpha)
        {
            if(tid < n_mod_8)
            {
                auto tx = load_ptr_batch(x, batch, offset_x + tid, stride_x);
                auto ty = load_ptr_batch(y, batch, offset_y + tid, stride_y);
                *ty += alpha * (*tx);
            }
        }

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

//!
//! @brief Optimized kernel for the groups of 8 half floating points.
//!
template <rocblas_int NB, typename Ta, typename Tx, typename Ty>
ROCBLAS_KERNEL(NB)
rocblas_haxpy_mlt_8_kernel(rocblas_int    n_mlt_8,
                           Ta             alpha_device_host,
                           rocblas_stride stride_alpha,
                           Tx             x,
                           rocblas_stride offset_x,
                           rocblas_stride stride_x,
                           Ty             y,
                           rocblas_stride offset_y,
                           rocblas_stride stride_y,
                           rocblas_int    batch_count)
{
    int64_t t8id = threadIdx.x + blockIdx.x * NB;

    uint32_t batch = blockIdx.z;
#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        // Load alpha into both sides of a rocblas_half2 for fma instructions.
        auto alpha_value = load_scalar(alpha_device_host, batch, stride_alpha);
        union
        {
            rocblas_half2 value;
            uint32_t      data;
        } alpha_h2 = {{alpha_value, alpha_value}};

        if(alpha_h2.data & 0x7fff)
        {
            rocblas_half2 y0, y1, y2, y3;
            rocblas_half2 x0, x1, x2, x3;
            rocblas_half2 z0, z1, z2, z3;

            auto tid = t8id * 8;
            if(tid < n_mlt_8)
            {
                //
                // Cast to rocblas_half8.
                // The reason rocblas_half8 does not appear in the signature
                // is due to the generalization of the non-batched/batched/strided batched case.
                // But the purpose of this routine is to specifically doing calculation with rocblas_half8 but also being general.
                // Then we can consider it is acceptable.
                //
                const rocblas_half8* ax
                    = (const rocblas_half8*)load_ptr_batch(x, batch, offset_x + tid, stride_x);
                rocblas_half8* ay
                    = (rocblas_half8*)load_ptr_batch(y, batch, offset_y + tid, stride_y);

                y0[0] = (*ay)[0];
                y0[1] = (*ay)[1];
                y1[0] = (*ay)[2];
                y1[1] = (*ay)[3];
                y2[0] = (*ay)[4];
                y2[1] = (*ay)[5];
                y3[0] = (*ay)[6];
                y3[1] = (*ay)[7];

                x0[0] = (*ax)[0];
                x0[1] = (*ax)[1];
                x1[0] = (*ax)[2];
                x1[1] = (*ax)[3];
                x2[0] = (*ax)[4];
                x2[1] = (*ax)[5];
                x3[0] = (*ax)[6];
                x3[1] = (*ax)[7];

                z0 = rocblas_fmadd_half2(alpha_h2.value, x0, y0);
                z1 = rocblas_fmadd_half2(alpha_h2.value, x1, y1);
                z2 = rocblas_fmadd_half2(alpha_h2.value, x2, y2);
                z3 = rocblas_fmadd_half2(alpha_h2.value, x3, y3);

                (*ay)[0] = z0[0];
                (*ay)[1] = z0[1];
                (*ay)[2] = z1[0];
                (*ay)[3] = z1[1];
                (*ay)[4] = z2[0];
                (*ay)[5] = z2[1];
                (*ay)[6] = z3[0];
                (*ay)[7] = z3[1];
            }
        }
#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

//!
//! @brief General template to compute y = a * x + y.
//!
template <typename API_INT, int NB, typename Tex, typename Ta, typename Tx, typename Ty>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_axpy_launcher(rocblas_handle handle,
                                   API_INT        n,
                                   Ta             alpha,
                                   rocblas_stride stride_alpha,
                                   Tx             x,
                                   rocblas_stride offset_x,
                                   API_INT        incx,
                                   rocblas_stride stride_x,
                                   Ty             y,
                                   rocblas_stride offset_y,
                                   API_INT        incy,
                                   rocblas_stride stride_y,
                                   API_INT        batch_count)
{
    if(n <= 0 || batch_count <= 0) // Quick return if possible. Not Argument error
    {
        return rocblas_status_success;
    }

    // Using rocblas_half ?
    static constexpr bool using_rocblas_half
        //cppcheck-suppress duplicateExpression
        = std::is_same_v<Ta, rocblas_half> && std::is_same_v<Tex, rocblas_half>;

    // Using float ?
    static constexpr bool using_rocblas_float
        = std::is_same_v<Ty, rocblas_float*> || std::is_same_v<Ty, rocblas_float* const*>;

    static constexpr rocblas_stride stride_0 = 0;

    //  unit_inc is True only if incx == 1  && incy == 1.
    bool unit_inc = (incx == 1 && incy == 1);

    int batches = handle->getBatchGridDim((int)batch_count);

    if(using_rocblas_half && unit_inc)
    {
        //
        // Optimized version of rocblas_half, where incx == 1 and incy == 1.
        // TODO: always use an optimized version.
        //
        //
        // Note: Do not use pointer arithmetic with x and y when passing parameters.
        // The kernel will do the cast if needed.
        //
        rocblas_int n_mod_8 = n & 7; // n mod 8
        rocblas_int n_mlt_8 = n & ~(rocblas_int)7; // multiple of 8
        int         blocks  = (n / 8 - 1) / NB + 1;
        dim3        grid(blocks, 1, batches);
        dim3        threads(NB);
        if(handle->pointer_mode == rocblas_pointer_mode_device)
        {
            // clang-format off
            ROCBLAS_LAUNCH_KERNEL((rocblas_haxpy_mlt_8_kernel<NB>), grid, threads, 0, handle->get_stream(), n_mlt_8,
                               (const rocblas_half*)alpha, stride_alpha, x, offset_x, stride_x, y, offset_y, stride_y, batch_count);
            // clang-format on
            if(n_mod_8)
            {
                //
                // cleanup non-multiple of 8
                //
                // clang-format off
                ROCBLAS_LAUNCH_KERNEL((rocblas_haxpy_mod_8_kernel<NB>), dim3(1, batch_count), n_mod_8, 0, handle->get_stream(), n_mod_8,
                                    alpha, stride_alpha, x, n_mlt_8 + offset_x, stride_x, y, n_mlt_8 + offset_y, stride_y, batch_count);
                // clang-format on
            }
        }
        else
        {
            // Note: We do not support batched alpha on host.
            // clang-format off
            ROCBLAS_LAUNCH_KERNEL((rocblas_haxpy_mlt_8_kernel<NB>), grid, threads, 0, handle->get_stream(),
                                n_mlt_8,load_scalar((const rocblas_half*)alpha), stride_0, x, offset_x, stride_x, y, offset_y, stride_y, batch_count);
            // clang-format on

            if(n_mod_8)
            {
                // clang-format off
                ROCBLAS_LAUNCH_KERNEL((rocblas_haxpy_mod_8_kernel<NB>), dim3(1, batch_count), n_mod_8, 0, handle->get_stream(), n_mod_8,
                                   *alpha, stride_0, x, n_mlt_8 + offset_x, stride_x, y, n_mlt_8 + offset_y, stride_y, batch_count);
                // clang-format on
            }
        }
    }

    else if(using_rocblas_float && unit_inc && batch_count <= 8192)
    {
        // Optimized kernel for float Datatype when incx==1 && incy==1 && batch_count <= 8192
        dim3 blocks((n - 1) / (NB * 2) + 1, 1, batches);
        dim3 threads(NB);

        if(rocblas_pointer_mode_device == handle->pointer_mode)
        {
            // clang-format off
            ROCBLAS_LAUNCH_KERNEL((rocblas_saxpy_2_kernel<NB, Tex>), blocks, threads, 0, handle->get_stream(), n, alpha,
                               stride_alpha, x, offset_x, stride_x, y, offset_y, stride_y, batch_count);
            // clang-format on
        }

        else
        {
            // Note: We do not support batched alpha on host.
            // clang-format off
            ROCBLAS_LAUNCH_KERNEL((rocblas_saxpy_2_kernel<NB, Tex>), blocks, threads, 0, handle->get_stream(), n, *alpha,
                               stride_0, x, offset_x, stride_x, y, offset_y, stride_y, batch_count);
            // clang-format on
        }
    }

    else if(batch_count > 8192 && using_rocblas_float)
    {
        // Optimized kernel for float Datatype when batch_count > 8192
        int64_t shift_x = offset_x + ((incx < 0) ? int64_t(incx) * (1 - n) : 0);
        int64_t shift_y = offset_y + ((incy < 0) ? int64_t(incy) * (1 - n) : 0);

        constexpr int DIM_X = 128;
        constexpr int DIM_Y = 8;

        dim3 blocks((n - 1) / DIM_X + 1, 1, (batches - 1) / (DIM_Y * 4) + 1);
        dim3 threads(DIM_X, DIM_Y);

        if(handle->pointer_mode == rocblas_pointer_mode_device)
        {
            // clang-format off
            ROCBLAS_LAUNCH_KERNEL((rocblas_axpy_kernel_batched<API_INT, DIM_X, DIM_Y, Tex>), blocks, threads, 0, handle->get_stream(), n, alpha,
                               stride_alpha, x, shift_x, incx, stride_x, y, shift_y, incy, stride_y, batch_count);
            // clang-format on
        }
        else
        {
            // Note: We do not support batched alpha on host.
            // clang-format off
            ROCBLAS_LAUNCH_KERNEL((rocblas_axpy_kernel_batched<API_INT, DIM_X, DIM_Y, Tex>), blocks, threads, 0, handle->get_stream(), n, *alpha,
                               stride_0, x, shift_x, incx, stride_x, y, shift_y, incy, stride_y, batch_count);
            // clang-format on
        }
    }

    else
    {
        // Default kernel for AXPY
        int64_t shift_x = offset_x + ((incx < 0) ? int64_t(incx) * (1 - n) : 0);
        int64_t shift_y = offset_y + ((incy < 0) ? int64_t(incy) * (1 - n) : 0);

        dim3 blocks((n - 1) / NB + 1, 1, batches);
        dim3 threads(NB);
        if(handle->pointer_mode == rocblas_pointer_mode_device)
        {
            // clang-format off
            ROCBLAS_LAUNCH_KERNEL((rocblas_axpy_kernel<API_INT, NB, Tex>), blocks, threads, 0, handle->get_stream(), n, alpha,
                               stride_alpha, x, shift_x, incx, stride_x, y,shift_y, incy, stride_y, batch_count);
            // clang-format on
        }
        else
        {
            // Note: We do not support batched alpha on host.
            // clang-format off
            ROCBLAS_LAUNCH_KERNEL((rocblas_axpy_kernel<API_INT, NB, Tex>), blocks, threads, 0, handle->get_stream(), n, *alpha,
                               stride_0, x, shift_x, incx, stride_x, y, shift_y, incy, stride_y, batch_count);
            // clang-format on
        }
    }
    return rocblas_status_success;
}

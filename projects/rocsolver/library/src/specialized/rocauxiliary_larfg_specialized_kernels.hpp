/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include "../auxiliary/rocauxiliary_larfg.hpp"
#include "lib_device_helpers.hpp"
#include "rocblas.hpp"
#include "rocsolver_run_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/*************************************************************
    Templated kernels are instantiated in separate cpp
    files in order to improve compilation times and reduce
    the library size.
*************************************************************/

template <int MAX_THDS, typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(MAX_THDS) larfg_kernel_small(const I n,
                                                                     U alpha,
                                                                     const rocblas_stride shiftA,
                                                                     const rocblas_stride strideA,
                                                                     S* beta,
                                                                     const rocblas_stride shiftB,
                                                                     const rocblas_stride strideB,
                                                                     U xx,
                                                                     const rocblas_stride shiftX,
                                                                     const I incX,
                                                                     const rocblas_stride strideX,
                                                                     T* tauA,
                                                                     const rocblas_stride strideP)
{
    I bid = blockIdx.z;
    I tid = threadIdx.x;

    // select batch instance
    T* a = load_ptr_batch<T>(alpha, bid, shiftA, strideA);
    T* x = load_ptr_batch<T>(xx, bid, shiftX, strideX);
    T* tau = load_ptr_batch<T>(tauA, bid, 0, strideP);

    S* b = beta ? load_ptr_batch<S>(beta, bid, shiftB, strideB) : nullptr;

    // shared variables
    __shared__ T sval[MAX_THDS / WarpSize];

    // dot
    T norm2 = 0;
    for(I i = tid; i < n - 1; i += MAX_THDS)
    {
        T temp = x[i * incX];
        norm2 += temp * conj(temp);
    }

    // reduce squared entries to find squared norm of x
    norm2 += shift_left(norm2, 1);
    norm2 += shift_left(norm2, 2);
    norm2 += shift_left(norm2, 4);
    norm2 += shift_left(norm2, 8);
    norm2 += shift_left(norm2, 16);
    if(warpSize > 32)
        norm2 += shift_left(norm2, 32);
    if(tid % warpSize == 0)
        sval[tid / warpSize] = norm2;
    __syncthreads();
    if(tid == 0)
    {
        for(I k = 1; k < MAX_THDS / warpSize; k++)
            norm2 += sval[k];
        sval[0] = norm2;
    }
    __syncthreads();

    // set tau, beta, and put scaling factor into sval[0]
    if(tid == 0)
    {
        run_set_taubeta<T>(tau, sval, a, b);
    }
    __syncthreads();

    // scale x by scaling factor
    for(I i = tid; i < n - 1; i += MAX_THDS)
        x[i * incX] *= sval[0];
}

/*************************************************************
    Launchers of specialized  kernels
*************************************************************/

template <typename T, typename I, typename S, typename U>
rocblas_status larfg_run_small(rocblas_handle handle,
                               const I n,
                               U alpha,
                               const rocblas_stride shiftA,
                               const rocblas_stride strideA,
                               S* beta,
                               const rocblas_stride shiftB,
                               const rocblas_stride strideB,
                               U x,
                               const rocblas_stride shiftX,
                               const I incX,
                               const rocblas_stride strideX,
                               T* tau,
                               const rocblas_stride strideP,
                               const I batch_count)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    if(n <= 64)
    {
        ROCSOLVER_LAUNCH_KERNEL((larfg_kernel_small<64, T>), dim3(1, 1, batch_count), dim3(64), 0,
                                stream, n, alpha, shiftA, strideA, beta, shiftB, strideB, x, shiftX,
                                incX, strideX, tau, strideP);
    }
    else if(n <= 128)
    {
        ROCSOLVER_LAUNCH_KERNEL((larfg_kernel_small<128, T>), dim3(1, 1, batch_count), dim3(128), 0,
                                stream, n, alpha, shiftA, strideA, beta, shiftB, strideB, x, shiftX,
                                incX, strideX, tau, strideP);
    }
    else if(n <= 256)
    {
        ROCSOLVER_LAUNCH_KERNEL((larfg_kernel_small<256, T>), dim3(1, 1, batch_count), dim3(256), 0,
                                stream, n, alpha, shiftA, strideA, beta, shiftB, strideB, x, shiftX,
                                incX, strideX, tau, strideP);
    }
    else if(n <= 512)
    {
        ROCSOLVER_LAUNCH_KERNEL((larfg_kernel_small<512, T>), dim3(1, 1, batch_count), dim3(512), 0,
                                stream, n, alpha, shiftA, strideA, beta, shiftB, strideB, x, shiftX,
                                incX, strideX, tau, strideP);
    }
    else
    {
        ROCSOLVER_LAUNCH_KERNEL((larfg_kernel_small<1024, T>), dim3(1, 1, batch_count), dim3(1024),
                                0, stream, n, alpha, shiftA, strideA, beta, shiftB, strideB, x,
                                shiftX, incX, strideX, tau, strideP);
    }

    return rocblas_status_success;
}

/*************************************************************
    Instantiation macros
*************************************************************/

#define INSTANTIATE_LARFG_SMALL(T, I, S, U)                                           \
    template rocblas_status larfg_run_small<T, I, S, U>(                              \
        rocblas_handle handle, const I n, U alpha, const rocblas_stride shiftA,       \
        const rocblas_stride strideA, S* beta, const rocblas_stride shiftB,           \
        const rocblas_stride strideB, U x, const rocblas_stride shiftX, const I incX, \
        const rocblas_stride strideX, T* tau, const rocblas_stride strideP, const I batch_count)

ROCSOLVER_END_NAMESPACE

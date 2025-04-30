/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "lapack_device_functions.hpp"
#include "rocsolver_run_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/*************************************************************
    Templated kernels are instantiated in separate cpp
    files in order to improve compilation times and reduce
    the library size.
*************************************************************/

template <typename T, typename I, typename U, typename UB>
ROCSOLVER_KERNEL void __launch_bounds__(LARFG_SSKER_THREADS)
    larfg_kernel_small(const I n,
                       U alpha,
                       const rocblas_stride shiftA,
                       const rocblas_stride strideA,
                       UB beta,
                       const rocblas_stride shiftB,
                       const rocblas_stride strideB,
                       U xx,
                       const rocblas_stride shiftX,
                       const I incX,
                       const rocblas_stride strideX,
                       T* tauA,
                       const rocblas_stride strideP)
{
    I bid = hipBlockIdx_x;
    I tid = hipThreadIdx_x;

    // select batch instance
    T* a = load_ptr_batch<T>(alpha, bid, shiftA, strideA);
    T* x = load_ptr_batch<T>(xx, bid, shiftX, strideX);
    T* tau = load_ptr_batch<T>(tauA, bid, 0, strideP);

    T* b = beta ? load_ptr_batch<T>(beta, bid, shiftB, strideB) : nullptr;

    // shared variables
    __shared__ T sval[LARFG_SSKER_THREADS];
    __shared__ T sh_x[LARFG_SSKER_MAX_N];

    // load x into shared memory
    for(I i = tid; i < n - 1; i += LARFG_SSKER_THREADS)
        sh_x[i] = x[i * incX];

    // find squared norm of x
    dot<LARFG_SSKER_THREADS, true, T>(tid, n - 1, sh_x, 1, sh_x, 1, sval);

    // set tau, beta, and put scaling factor into sval[0]
    if(tid == 0)
        run_set_taubeta<T>(tau, sval, a, b);
    __syncthreads();

    // scale x by scaling factor
    for(I i = tid; i < n - 1; i += LARFG_SSKER_THREADS)
        x[i * incX] = sh_x[i] * sval[0];
}

/*************************************************************
    Launchers of specialized  kernels
*************************************************************/

template <typename T, typename I, typename U>
rocblas_status larfg_run_small(rocblas_handle handle,
                               const I n,
                               U alpha,
                               const rocblas_stride shiftA,
                               const rocblas_stride strideA,
                               T* beta,
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
    dim3 grid(batch_count, 1, 1);
    dim3 block(LARFG_SSKER_THREADS, 1, 1);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    ROCSOLVER_LAUNCH_KERNEL((larfg_kernel_small<T>), grid, block, 0, stream, n, alpha, shiftA,
                            strideA, beta, shiftB, strideB, x, shiftX, incX, strideX, tau, strideP);

    return rocblas_status_success;
}

/*************************************************************
    Instantiation macros
*************************************************************/

#define INSTANTIATE_LARFG_SMALL(T, I, U)                                              \
    template rocblas_status larfg_run_small<T, I, U>(                                 \
        rocblas_handle handle, const I n, U alpha, const rocblas_stride shiftA,       \
        const rocblas_stride strideA, T* beta, const rocblas_stride shiftB,           \
        const rocblas_stride strideB, U x, const rocblas_stride shiftX, const I incX, \
        const rocblas_stride strideX, T* tau, const rocblas_stride strideP, const I batch_count)

ROCSOLVER_END_NAMESPACE

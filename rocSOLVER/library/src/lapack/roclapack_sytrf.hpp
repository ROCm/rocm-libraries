/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../auxiliary/rocauxiliary_lasyf.hpp"
#include "rocblas.hpp"
#include "roclapack_sytf2.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

/** thread-block size for calling the sytrf kernel.
    (MAX_THDS sizes must be one of 128, 256, 512, or 1024) **/
#define SYTRF_MAX_THDS 256

template <typename T, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(SYTRF_MAX_THDS)
    sytrf_kernel_upper(const rocblas_int n,
                       U AA,
                       const rocblas_int shiftA,
                       const rocblas_int lda,
                       const rocblas_stride strideA,
                       rocblas_int* ipivA,
                       const rocblas_stride strideP,
                       rocblas_int* info,
                       T* WA)
{
    using S = decltype(std::real(T{}));

    // select batch instance
    rocblas_int bid = hipBlockIdx_y;
    rocblas_int tid = hipThreadIdx_x;

    // get array pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* W = WA + (bid * n * SYTRF_MAX_THDS);
    rocblas_int* ipiv = ipivA + (bid * strideP);

    // local and shared variables
    __shared__ int iinfo;
    __shared__ int kb;
    int k = n - 1;

    // shared arrays
    __shared__ S sval[SYTRF_MAX_THDS];
    __shared__ rocblas_int sidx[SYTRF_MAX_THDS];

    if(tid == 0)
        info[bid] = 0;

    while(k >= 0)
    {
        if(k >= SYTRF_SYTF2_SWITCHSIZE)
        {
            lasyf_device_upper<SYTRF_MAX_THDS>(tid, k + 1, SYTRF_BLOCKSIZE, &kb, A, lda, ipiv,
                                               &iinfo, W, sidx, sval);
            k = k - kb;
        }
        else
        {
            sytf2_device_upper<SYTRF_MAX_THDS>(tid, k + 1, A, lda, ipiv, &iinfo, sidx, sval);
            k = -1;
        }

        if(tid == 0 && iinfo != 0 && info[bid] == 0)
            info[bid] = iinfo;
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(SYTRF_MAX_THDS)
    sytrf_kernel_lower(const rocblas_int n,
                       U AA,
                       const rocblas_int shiftA,
                       const rocblas_int lda,
                       const rocblas_stride strideA,
                       rocblas_int* ipivA,
                       const rocblas_stride strideP,
                       rocblas_int* info,
                       T* WA)
{
    using S = decltype(std::real(T{}));

    // select batch instance
    rocblas_int bid = hipBlockIdx_y;
    rocblas_int tid = hipThreadIdx_x;

    // get array pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* W = WA + (bid * n * SYTRF_MAX_THDS);
    rocblas_int* ipiv = ipivA + (bid * strideP);

    // local and shared variables
    __shared__ int iinfo;
    __shared__ int kb;
    int k = 0;
    int ktemp, j;

    // shared arrays
    __shared__ S sval[SYTRF_MAX_THDS];
    __shared__ rocblas_int sidx[SYTRF_MAX_THDS];

    if(tid == 0)
        info[bid] = 0;

    while(k < n)
    {
        if(k < n - SYTRF_SYTF2_SWITCHSIZE)
        {
            lasyf_device_lower<SYTRF_MAX_THDS>(tid, n - k, SYTRF_BLOCKSIZE, &kb, A + k + k * lda,
                                               lda, ipiv + k, &iinfo, W, sidx, sval);
            ktemp = k + kb;
        }
        else
        {
            sytf2_device_lower<SYTRF_MAX_THDS>(tid, n - k, A + k + k * lda, lda, ipiv + k, &iinfo,
                                               sidx, sval);
            ktemp = n;
            __syncthreads();
        }

        if(tid == 0 && iinfo != 0 && info[bid] == 0)
            info[bid] = iinfo + k;

        // adjust pivots
        for(j = k + tid; j < ktemp; j += SYTRF_MAX_THDS)
        {
            if(ipiv[j] > 0)
                ipiv[j] += k;
            else
                ipiv[j] -= k;
        }

        k = ktemp;
    }
}

template <typename T>
void rocsolver_sytrf_getMemorySize(const rocblas_int n, const rocblas_int batch_count, size_t* size_work)
{
    // if quick return no workspace needed
    if(n == 0 || batch_count == 0)
    {
        *size_work = 0;
        return;
    }

    // size of workspace
    if(n > SYTRF_SYTF2_SWITCHSIZE)
        rocsolver_lasyf_getMemorySize<T>(n, SYTRF_BLOCKSIZE, batch_count, size_work);
    else
        *size_work = 0;
}

template <typename T, typename U>
rocblas_status rocsolver_sytrf_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        rocblas_int* ipiv,
                                        const rocblas_stride strideP,
                                        rocblas_int* info,
                                        const rocblas_int batch_count,
                                        T* work)
{
    ROCSOLVER_ENTER("sytrf", "uplo:", uplo, "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // quick return
    if(n == 0)
    {
        // set info = 0
        rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
        dim3 gridReset(blocksReset, 1, 1);
        dim3 threadsReset(BS1, 1, 1);
        ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threadsReset, 0, stream, info, batch_count, 0);

        return rocblas_status_success;
    }

    dim3 grid(1, batch_count, 1);
    dim3 threads(SYTRF_MAX_THDS, 1, 1);

    if(uplo == rocblas_fill_upper)
        ROCSOLVER_LAUNCH_KERNEL(sytrf_kernel_upper<T>, grid, threads, 0, stream, n, A, shiftA, lda,
                                strideA, ipiv, strideP, info, work);
    else
        ROCSOLVER_LAUNCH_KERNEL(sytrf_kernel_lower<T>, grid, threads, 0, stream, n, A, shiftA, lda,
                                strideA, ipiv, strideP, info, work);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE

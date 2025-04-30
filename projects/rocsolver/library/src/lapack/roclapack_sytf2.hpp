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

#include "lapack_device_functions.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

/** thread-block size for calling the sytf2 kernel.
    (MAX_THDS sizes must be one of 128, 256, 512, or 1024) **/
#define SYTF2_MAX_THDS 256

template <int MAX_THDS, typename T, typename S>
__device__ void sytf2_device_upper(const rocblas_int tid,
                                   const rocblas_int n,
                                   T* A,
                                   const rocblas_int lda,
                                   rocblas_int* ipiv,
                                   rocblas_int* info,
                                   rocblas_int* sidx,
                                   S* sval)
{
    const S alpha = S((1.0 + std::sqrt(17.0)) / 8.0);

    // local and shared variables
    __shared__ rocblas_int _info;
    int i, j;
    int k = n - 1;
    int kp, kk;

    // shared variables for iamax
    __shared__ S absakk;
    __shared__ S colmax;
    __shared__ S rowmax;
    __shared__ rocblas_int imax;

    if(tid == 0)
        _info = 0;

    while(k >= 0)
    {
        int kstep = 1;

        // find max off-diagonal entry in column k
        iamax<MAX_THDS>(tid, k, A + k * lda, 1, sval, sidx);
        if(tid == 0)
        {
            imax = sidx[0] - 1;
            colmax = sval[0];
            absakk = aabs<S>(A[k + k * lda]);
        }
        __syncthreads();

        if(std::max(absakk, colmax) == 0)
        {
            // singularity found
            if(tid == 0 && _info == 0)
                _info = k + 1;
            kp = k;
        }
        else
        {
            if(absakk >= alpha * colmax)
                // no interchange (1-by-1 block)
                kp = k;
            else
            {
                // find max off-diagonal entry in row i
                iamax<MAX_THDS>(tid, k - imax, A + imax + (imax + 1) * lda, lda, sval, sidx);
                if(tid == 0)
                    rowmax = sval[0];

                if(imax > 0)
                {
                    iamax<MAX_THDS>(tid, imax, A + imax * lda, 1, sval, sidx);
                    if(tid == 0)
                        rowmax = std::max(rowmax, sval[0]);
                }
                __syncthreads();

                if(absakk >= alpha * colmax * (colmax / rowmax))
                    // no interchange (1-by-1 block)
                    kp = k;
                else if(aabs<S>(A[imax + imax * lda]) >= alpha * rowmax)
                    // interchange rows and columns kk = k and kp = imax (1-by-1 block)
                    kp = imax;
                else
                {
                    // interchange rows and columns kk = k-1 and kp = imax (2-by-2 block)
                    kp = imax;
                    kstep = 2;
                }
            }

            kk = k - kstep + 1;
            if(kp != kk)
            {
                // interchange rows and columns kp and kk
                if(tid == 0)
                {
                    swap(A[kk + kk * lda], A[kp + kp * lda]);
                    if(kstep == 2)
                        swap(A[kk + k * lda], A[kp + k * lda]);
                }

                for(i = tid; i < kp; i += MAX_THDS)
                    swap(A[i + kk * lda], A[i + kp * lda]);
                for(i = tid; i < kk - kp - 1; i += MAX_THDS)
                    swap(A[(kp + i + 1) + kk * lda], A[kp + (kp + i + 1) * lda]);
                __syncthreads();
            }

            if(kstep == 1)
            {
                // 1-by-1 pivot block

                // perform rank 1 update of A from [0,0] to [k-1,k-1] (syr)
                T r1 = T(1) / A[k + k * lda];
                for(j = tid; j < k; j += MAX_THDS)
                {
                    T r2 = -r1 * A[j + k * lda];
                    for(i = 0; i <= j; i++)
                        A[i + j * lda] = A[i + j * lda] + A[i + k * lda] * r2;
                }
                __syncthreads();

                // update column k (scal)
                for(j = tid; j < k; j += MAX_THDS)
                    A[j + k * lda] *= r1;
            }
            else
            {
                // 2-by-2 pivot block

                if(k > 1)
                {
                    // perform rank 2 update of A from [0,0] to [k-2,k-2]
                    T wk, wkm1;
                    T d12 = A[(k - 1) + k * lda];
                    T d22 = A[(k - 1) + (k - 1) * lda] / d12;
                    T d11 = A[k + k * lda] / d12;
                    d12 = T(1) / ((d11 * d22 - T(1)) * d12);
                    for(j = k - 2 - tid; j >= 0; j -= MAX_THDS)
                    {
                        wkm1 = d12 * (d11 * A[j + (k - 1) * lda] - A[j + k * lda]);
                        wk = d12 * (d22 * A[j + k * lda] - A[j + (k - 1) * lda]);
                        for(i = j; i >= 0; i--)
                            A[i + j * lda]
                                = A[i + j * lda] - A[i + k * lda] * wk - A[i + (k - 1) * lda] * wkm1;
                    }
                    __syncthreads();

                    // update columns k and k-1
                    for(j = k - 2 - tid; j >= 0; j -= MAX_THDS)
                    {
                        wkm1 = d12 * (d11 * A[j + (k - 1) * lda] - A[j + k * lda]);
                        wk = d12 * (d22 * A[j + k * lda] - A[j + (k - 1) * lda]);
                        A[j + k * lda] = wk;
                        A[j + (k - 1) * lda] = wkm1;
                    }
                }
            }
        }

        // update ipiv (1-based index to match LAPACK)
        if(tid == 0)
        {
            if(kstep == 1)
                ipiv[k] = kp + 1;
            else
            {
                ipiv[k] = -(kp + 1);
                ipiv[k - 1] = -(kp + 1);
            }
        }

        k -= kstep;
    }

    if(tid == 0)
        *info = _info;
}

template <int MAX_THDS, typename T, typename S>
__device__ void sytf2_device_lower(const rocblas_int tid,
                                   const rocblas_int n,
                                   T* A,
                                   const rocblas_int lda,
                                   rocblas_int* ipiv,
                                   rocblas_int* info,
                                   rocblas_int* sidx,
                                   S* sval)
{
    const S alpha = S((1.0 + std::sqrt(17.0)) / 8.0);

    // local and shared variables
    __shared__ rocblas_int _info;
    int i, j;
    int k = 0;
    int kp, kk;

    // shared variables for iamax
    __shared__ S absakk;
    __shared__ S colmax;
    __shared__ S rowmax;
    __shared__ rocblas_int imax;

    if(tid == 0)
        _info = 0;

    while(k < n)
    {
        int kstep = 1;

        // find max off-diagonal entry in column k
        iamax<MAX_THDS>(tid, n - k - 1, A + (k + 1) + k * lda, 1, sval, sidx);
        if(tid == 0)
        {
            imax = k + sidx[0];
            colmax = sval[0];
            absakk = aabs<S>(A[k + k * lda]);
        }
        __syncthreads();

        if(std::max(absakk, colmax) == 0)
        {
            // singularity found
            if(tid == 0 && _info == 0)
                _info = k + 1;
            kp = k;
        }
        else
        {
            if(absakk >= alpha * colmax)
                // no interchange (1-by-1 block)
                kp = k;
            else
            {
                // find max off-diagonal entry in row i
                iamax<MAX_THDS>(tid, imax - k, A + imax + k * lda, lda, sval, sidx);
                if(tid == 0)
                    rowmax = sval[0];

                if(imax < n - 1)
                {
                    iamax<MAX_THDS>(tid, n - imax - 1, A + (imax + 1) + imax * lda, 1, sval, sidx);
                    if(tid == 0)
                        rowmax = std::max(rowmax, sval[0]);
                }
                __syncthreads();

                if(absakk >= alpha * colmax * (colmax / rowmax))
                    // no interchange (1-by-1 block)
                    kp = k;
                else if(aabs<S>(A[imax + imax * lda]) >= alpha * rowmax)
                    // interchange rows and columns kk = k and kp = imax (1-by-1 block)
                    kp = imax;
                else
                {
                    // interchange rows and columns kk = k+1 and kp = imax (2-by-2 block)
                    kp = imax;
                    kstep = 2;
                }
            }

            kk = k + kstep - 1;
            if(kp != kk)
            {
                // interchange rows and columns kp and kk
                if(tid == 0)
                {
                    swap(A[kk + kk * lda], A[kp + kp * lda]);
                    if(kstep == 2)
                        swap(A[kk + k * lda], A[kp + k * lda]);
                }

                for(i = tid; i < n - kp - 1; i += MAX_THDS)
                    swap(A[(kp + i + 1) + kk * lda], A[(kp + i + 1) + kp * lda]);
                for(i = tid; i < kp - kk - 1; i += MAX_THDS)
                    swap(A[(kk + i + 1) + kk * lda], A[kp + (kk + i + 1) * lda]);
                __syncthreads();
            }

            if(kstep == 1)
            {
                // 1-by-1 pivot block

                if(k < n - 1)
                {
                    // perform rank 1 update of A from [k+1,k+1] to [n-1,n-1] (syr)
                    T r1 = T(1) / A[k + k * lda];
                    for(j = tid; j < n - k - 1; j += MAX_THDS)
                    {
                        T r2 = -r1 * A[(k + j + 1) + k * lda];
                        for(i = j; i < n - k - 1; i++)
                            A[(k + i + 1) + (k + j + 1) * lda]
                                = A[(k + i + 1) + (k + j + 1) * lda] + A[(k + i + 1) + k * lda] * r2;
                    }
                    __syncthreads();

                    // update column k (scal)
                    for(j = tid; j < n - k - 1; j += MAX_THDS)
                        A[(k + j + 1) + k * lda] *= r1;
                }
            }
            else
            {
                // 2-by-2 pivot block

                if(k < n - 2)
                {
                    // perform rank 2 update of A from [k+2,k+2] to [n-1,n-1]
                    T wk, wkp1;
                    T d21 = A[(k + 1) + k * lda];
                    T d11 = A[(k + 1) + (k + 1) * lda] / d21;
                    T d22 = A[k + k * lda] / d21;
                    d21 = T(1) / ((d11 * d22 - T(1)) * d21);
                    for(j = k + 2 + tid; j < n; j += MAX_THDS)
                    {
                        wk = d21 * (d11 * A[j + k * lda] - A[j + (k + 1) * lda]);
                        wkp1 = d21 * (d22 * A[j + (k + 1) * lda] - A[j + k * lda]);
                        for(i = j; i < n; i++)
                            A[i + j * lda]
                                = A[i + j * lda] - A[i + k * lda] * wk - A[i + (k + 1) * lda] * wkp1;
                    }
                    __syncthreads();

                    // update columns k and k+1
                    for(j = k + 2 + tid; j < n; j += MAX_THDS)
                    {
                        wk = d21 * (d11 * A[j + k * lda] - A[j + (k + 1) * lda]);
                        wkp1 = d21 * (d22 * A[j + (k + 1) * lda] - A[j + k * lda]);
                        A[j + k * lda] = wk;
                        A[j + (k + 1) * lda] = wkp1;
                    }
                }
            }
        }

        // update ipiv (1-based index to match LAPACK)
        if(tid == 0)
        {
            if(kstep == 1)
                ipiv[k] = kp + 1;
            else
            {
                ipiv[k] = -(kp + 1);
                ipiv[k + 1] = -(kp + 1);
            }
        }

        k += kstep;
    }

    if(tid == 0)
        *info = _info;
}

template <typename T, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(SYTF2_MAX_THDS)
    sytf2_kernel_upper(const rocblas_int n,
                       U AA,
                       const rocblas_int shiftA,
                       const rocblas_int lda,
                       const rocblas_stride strideA,
                       rocblas_int* ipivA,
                       const rocblas_stride strideP,
                       rocblas_int* infoA)
{
    using S = decltype(std::real(T{}));

    // select batch instance
    rocblas_int bid = hipBlockIdx_y;
    rocblas_int tid = hipThreadIdx_x;

    // get array pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    rocblas_int* ipiv = ipivA + (bid * strideP);

    // shared arrays
    __shared__ S sval[SYTF2_MAX_THDS];
    __shared__ rocblas_int sidx[SYTF2_MAX_THDS];

    sytf2_device_upper<SYTF2_MAX_THDS>(tid, n, A, lda, ipiv, infoA + bid, sidx, sval);
}

template <typename T, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(SYTF2_MAX_THDS)
    sytf2_kernel_lower(const rocblas_int n,
                       U AA,
                       const rocblas_int shiftA,
                       const rocblas_int lda,
                       const rocblas_stride strideA,
                       rocblas_int* ipivA,
                       const rocblas_stride strideP,
                       rocblas_int* infoA)
{
    using S = decltype(std::real(T{}));

    // select batch instance
    rocblas_int bid = hipBlockIdx_y;
    rocblas_int tid = hipThreadIdx_x;

    // get array pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    rocblas_int* ipiv = ipivA + (bid * strideP);

    // shared arrays
    __shared__ S sval[SYTF2_MAX_THDS];
    __shared__ rocblas_int sidx[SYTF2_MAX_THDS];

    sytf2_device_lower<SYTF2_MAX_THDS>(tid, n, A, lda, ipiv, infoA + bid, sidx, sval);
}

template <typename T>
rocblas_status rocsolver_sytf2_sytrf_argCheck(rocblas_handle handle,
                                              const rocblas_fill uplo,
                                              const rocblas_int n,
                                              const rocblas_int lda,
                                              T A,
                                              rocblas_int* ipiv,
                                              rocblas_int* info,
                                              const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !ipiv) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U>
rocblas_status rocsolver_sytf2_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        U A,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        rocblas_int* ipiv,
                                        const rocblas_stride strideP,
                                        rocblas_int* info,
                                        const rocblas_int batch_count)
{
    ROCSOLVER_ENTER("sytf2", "uplo:", uplo, "n:", n, "shiftA:", shiftA, "lda:", lda,
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
    dim3 threads(SYTF2_MAX_THDS, 1, 1);

    if(uplo == rocblas_fill_upper)
        ROCSOLVER_LAUNCH_KERNEL(sytf2_kernel_upper<T>, grid, threads, 0, stream, n, A, shiftA, lda,
                                strideA, ipiv, strideP, info);
    else
        ROCSOLVER_LAUNCH_KERNEL(sytf2_kernel_lower<T>, grid, threads, 0, stream, n, A, shiftA, lda,
                                strideA, ipiv, strideP, info);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE

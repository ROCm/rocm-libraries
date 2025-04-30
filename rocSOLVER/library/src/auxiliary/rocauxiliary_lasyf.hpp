/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.1) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     June 2017
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

/** thread-block size for calling the lasyf kernel.
    (MAX_THDS sizes must be one of 128, 256, 512, or 1024) **/
#define LASYF_MAX_THDS 256

/** GEMV device function to compute y = alpha * A * x + beta * y **/
template <int MAX_THDS, typename T>
__device__ void lasyf_gemv(const rocblas_int tid,
                           const rocblas_int m,
                           const rocblas_int n,
                           const T alpha,
                           T* A,
                           const rocblas_int lda,
                           T* x,
                           const rocblas_int incx,
                           const T beta,
                           T* y,
                           const rocblas_int incy)
{
    // gemv function assuming no transpose
    for(int i = tid; i < m; i += MAX_THDS)
    {
        T temp = 0;
        for(int j = 0; j < n; j++)
            temp += A[i + j * lda] * x[j * incx];
        y[i * incy] = alpha * temp + beta * y[i * incy];
    }
}

/** GEMM device function to compute C = alpha * A * B' + beta * C **/
template <int MAX_THDS, typename T>
__device__ void lasyf_gemm(const rocblas_int tid,
                           const rocblas_int m,
                           const rocblas_int n,
                           const rocblas_int k,
                           const T alpha,
                           T* A,
                           const rocblas_int lda,
                           T* B,
                           const rocblas_int ldb,
                           const T beta,
                           T* C,
                           const rocblas_int ldc)
{
    // gemm function assuming B transpose
    for(int e = tid; e < m * n; e += MAX_THDS)
    {
        int i = e % m;
        int j = e / m;
        T temp = 0;
        for(int l = 0; l < k; l++)
            temp += A[i + l * lda] * B[j + l * ldb];
        C[i + j * ldc] = alpha * temp + beta * C[i + j * ldc];
    }
}

template <int MAX_THDS, typename T, typename S>
__device__ void lasyf_device_upper(const rocblas_int tid,
                                   const rocblas_int n,
                                   const rocblas_int nb,
                                   rocblas_int* kb,
                                   T* A,
                                   const rocblas_int lda,
                                   rocblas_int* ipiv,
                                   rocblas_int* info,
                                   T* W,
                                   rocblas_int* sidx,
                                   S* sval)
{
    const S alpha = S((1.0 + std::sqrt(17.0)) / 8.0);
    const T one = 1;
    const T minone = -1;
    const int ldw = n;

    // local and shared variables
    __shared__ rocblas_int _info;
    int i, j;
    int k = n - 1;
    int kp, kk, kw, kkw;

    // shared variables for iamax
    __shared__ S absakk;
    __shared__ S colmax;
    __shared__ S rowmax;
    __shared__ rocblas_int imax;

    if(tid == 0)
        _info = 0;

    kw = nb + k - n;
    while(k >= 0 && (k > n - nb || nb == n))
    {
        // copy column k of A to column kw of W and update
        for(i = tid; i <= k; i += MAX_THDS)
            W[i + kw * ldw] = A[i + k * lda];
        __syncthreads();
        if(k < n - 1)
        {
            lasyf_gemv<MAX_THDS>(tid, k + 1, n - k - 1, minone, A + (k + 1) * lda, lda,
                                 W + k + (kw + 1) * ldw, ldw, one, W + kw * ldw, 1);
            __syncthreads();
        }

        int kstep = 1;

        // find max off-diagonal entry in column k
        iamax<MAX_THDS>(tid, k, W + kw * ldw, 1, sval, sidx);
        if(tid == 0)
        {
            imax = sidx[0] - 1;
            colmax = sval[0];
            absakk = aabs<S>(W[k + kw * ldw]);
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
                // copy column imax of A to column kw-1 of W and update
                for(i = tid; i <= imax; i += MAX_THDS)
                    W[i + (kw - 1) * ldw] = A[i + imax * lda];
                for(i = tid; i < k - imax; i += MAX_THDS)
                    W[(imax + i + 1) + (kw - 1) * ldw] = A[imax + (imax + i + 1) * lda];
                __syncthreads();
                if(k < n - 1)
                {
                    lasyf_gemv<MAX_THDS>(tid, k + 1, n - k - 1, minone, A + (k + 1) * lda, lda,
                                         W + imax + (kw + 1) * ldw, ldw, one, W + (kw - 1) * ldw, 1);
                    __syncthreads();
                }

                // find max off-diagonal entry in row imax
                iamax<MAX_THDS>(tid, k - imax, W + (imax + 1) + (kw - 1) * ldw, 1, sval, sidx);
                if(tid == 0)
                    rowmax = sval[0];

                if(imax > 0)
                {
                    iamax<MAX_THDS>(tid, imax, W + (kw - 1) * ldw, 1, sval, sidx);
                    if(tid == 0)
                        rowmax = std::max(rowmax, sval[0]);
                }
                __syncthreads();

                if(absakk >= alpha * colmax * (colmax / rowmax))
                    // no interchange (1-by-1 block)
                    kp = k;
                else if(aabs<S>(W[imax + (kw - 1) * ldw]) >= alpha * rowmax)
                {
                    // interchange rows and columns kk = k and kp = imax (1-by-1 block)
                    kp = imax;

                    // copy column kw-1 of W to column kw of W
                    for(i = tid; i <= k; i += MAX_THDS)
                        W[i + kw * ldw] = W[i + (kw - 1) * ldw];
                    __syncthreads();
                }
                else
                {
                    // interchange rows and columns kk = k-1 and kp = imax (2-by-2 block)
                    kp = imax;
                    kstep = 2;
                }
            }

            kk = k - kstep + 1;
            kkw = nb + kk - n;
            if(kp != kk)
            {
                // interchange rows and columns kp and kk
                if(tid == 0)
                    A[kp + kp * lda] = A[kk + kk * lda];

                for(i = tid; i < kk - kp - 1; i += MAX_THDS)
                    A[kp + (kp + i + 1) * lda] = A[(kp + i + 1) + kk * lda];
                for(i = tid; i < kp; i += MAX_THDS)
                    A[i + kp * lda] = A[i + kk * lda];
                __syncthreads();
                for(i = tid; i < n - k - 1; i += MAX_THDS)
                    swap(A[kk + (k + i + 1) * lda], A[kp + (k + i + 1) * lda]);
                for(i = tid; i < n - kk; i += MAX_THDS)
                    swap(W[kk + (kkw + i) * ldw], W[kp + (kkw + i) * ldw]);
                __syncthreads();
            }

            if(kstep == 1)
            {
                // 1-by-1 pivot block

                T r1 = T(1) / W[k + kw * ldw];
                if(tid == 0)
                    A[k + k * lda] = W[k + kw * ldw];
                for(i = tid; i < k; i += MAX_THDS)
                    A[i + k * lda] = r1 * W[i + kw * ldw];
                __syncthreads();
            }
            else
            {
                // 2-by-2 pivot block

                if(k > 1)
                {
                    T d21 = W[(k - 1) + kw * ldw];
                    T d11 = W[k + kw * ldw] / d21;
                    T d22 = W[(k - 1) + (kw - 1) * ldw] / d21;
                    d21 = T(1) / ((d11 * d22 - T(1)) * d21);
                    for(i = tid; i <= k - 2; i += MAX_THDS)
                    {
                        A[i + (k - 1) * lda] = d21 * (d11 * W[i + (kw - 1) * ldw] - W[i + kw * ldw]);
                        A[i + k * lda] = d21 * (d22 * W[i + kw * ldw] - W[i + (kw - 1) * ldw]);
                    }
                }

                if(tid == 0)
                {
                    A[(k - 1) + (k - 1) * lda] = W[(k - 1) + (kw - 1) * ldw];
                    A[(k - 1) + k * lda] = W[(k - 1) + kw * ldw];
                    A[k + k * lda] = W[k + kw * ldw];
                }
                __syncthreads();
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
        kw = nb + k - n;
    }

    if(tid == 0)
    {
        *kb = n - k - 1;
        *info = _info;
    }

    // update A from [0,0] to [k,k], nb columns at a time
    for(j = (k / nb) * nb; j >= 0; j -= nb)
    {
        int jb = std::min(nb, k - j + 1);
        for(i = j; i < j + jb; i++)
            lasyf_gemv<MAX_THDS>(tid, i - j + 1, n - k - 1, minone, A + j + (k + 1) * lda, lda,
                                 W + i + (kw + 1) * ldw, ldw, one, A + j + i * lda, 1);
        lasyf_gemm<MAX_THDS>(tid, j, jb, n - k - 1, minone, A + (k + 1) * lda, lda,
                             W + j + (kw + 1) * ldw, ldw, one, A + j * lda, lda);
    }
    __syncthreads();

    // partially undo interchanges to put U12 in standard form
    j = k + 1;
    while(j < n - 1)
    {
        kk = j; // jj
        kp = ipiv[j]; // jp
        if(kp < 0)
        {
            kp = -kp - 1;
            j++;
        }
        else
            kp = kp - 1;

        j++;
        if(kp != kk && j < n)
        {
            for(i = tid; i < n - j; i += MAX_THDS)
                swap(A[kp + (j + i) * lda], A[kk + (j + i) * lda]);
            __syncthreads();
        }
    }
}

template <int MAX_THDS, typename T, typename S>
__device__ void lasyf_device_lower(const rocblas_int tid,
                                   const rocblas_int n,
                                   const rocblas_int nb,
                                   rocblas_int* kb,
                                   T* A,
                                   const rocblas_int lda,
                                   rocblas_int* ipiv,
                                   rocblas_int* info,
                                   T* W,
                                   rocblas_int* sidx,
                                   S* sval)
{
    const S alpha = S((1.0 + std::sqrt(17.0)) / 8.0);
    const T one = 1;
    const T minone = -1;
    const int ldw = n;

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

    while(k < n && (k < nb - 1 || nb == n))
    {
        // copy column k of A to column k of W and update
        for(i = tid; i < n - k; i += MAX_THDS)
            W[(k + i) + k * ldw] = A[(k + i) + k * lda];
        __syncthreads();
        lasyf_gemv<MAX_THDS>(tid, n - k, k, minone, A + k, lda, W + k, ldw, one, W + k + k * ldw, 1);
        __syncthreads();

        int kstep = 1;

        // find max off-diagonal entry in column k
        iamax<MAX_THDS>(tid, n - k - 1, W + (k + 1) + k * ldw, 1, sval, sidx);
        if(tid == 0)
        {
            imax = k + sidx[0];
            colmax = sval[0];
            absakk = aabs<S>(W[k + k * ldw]);
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
                // copy column imax of A to column k+1 of W and update
                for(i = tid; i < imax - k; i += MAX_THDS)
                    W[(k + i) + (k + 1) * ldw] = A[imax + (k + i) * lda];
                for(i = tid; i < n - imax; i += MAX_THDS)
                    W[(imax + i) + (k + 1) * ldw] = A[(imax + i) + imax * lda];
                __syncthreads();
                lasyf_gemv<MAX_THDS>(tid, n - k, k, minone, A + k, lda, W + imax, ldw, one,
                                     W + k + (k + 1) * ldw, 1);
                __syncthreads();

                // find max off-diagonal entry in row imax
                iamax<MAX_THDS>(tid, imax - k, W + k + (k + 1) * ldw, 1, sval, sidx);
                if(tid == 0)
                    rowmax = sval[0];

                if(imax < n - 1)
                {
                    iamax<MAX_THDS>(tid, n - imax - 1, W + (imax + 1) + (k + 1) * ldw, 1, sval, sidx);
                    if(tid == 0)
                        rowmax = std::max(rowmax, sval[0]);
                }
                __syncthreads();

                if(absakk >= alpha * colmax * (colmax / rowmax))
                    // no interchange (1-by-1 block)
                    kp = k;
                else if(aabs<S>(W[imax + (k + 1) * ldw]) >= alpha * rowmax)
                {
                    // interchange rows and columns kk = k and kp = imax (1-by-1 block)
                    kp = imax;

                    // copy column kw-1 of W to column kw of W
                    for(i = tid; i < n - k; i += MAX_THDS)
                        W[(k + i) + k * ldw] = W[(k + i) + (k + 1) * ldw];
                    __syncthreads();
                }
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
                    A[kp + kp * lda] = A[kk + kk * lda];

                for(i = tid; i < kp - kk - 1; i += MAX_THDS)
                    A[kp + (kk + i + 1) * lda] = A[(kk + i + 1) + kk * lda];
                for(i = tid; i < n - kp - 1; i += MAX_THDS)
                    A[(kp + i + 1) + kp * lda] = A[(kp + i + 1) + kk * lda];
                __syncthreads();
                for(i = tid; i < k; i += MAX_THDS)
                    swap(A[kk + i * lda], A[kp + i * lda]);
                for(i = tid; i <= kk; i += MAX_THDS)
                    swap(W[kk + i * ldw], W[kp + i * ldw]);
                __syncthreads();
            }

            if(kstep == 1)
            {
                // 1-by-1 pivot block

                T r1 = T(1) / W[k + k * ldw];
                if(tid == 0)
                    A[k + k * lda] = W[k + k * ldw];
                for(i = tid; i < n - k - 1; i += MAX_THDS)
                    A[(k + i + 1) + k * lda] = r1 * W[(k + i + 1) + k * ldw];
                __syncthreads();
            }
            else
            {
                // 2-by-2 pivot block

                if(k < n - 2)
                {
                    T d21 = W[(k + 1) + k * ldw];
                    T d11 = W[(k + 1) + (k + 1) * ldw] / d21;
                    T d22 = W[k + k * ldw] / d21;
                    d21 = T(1) / ((d11 * d22 - T(1)) * d21);
                    for(i = k + 2 + tid; i < n; i += MAX_THDS)
                    {
                        A[i + k * lda] = d21 * (d11 * W[i + k * ldw] - W[i + (k + 1) * ldw]);
                        A[i + (k + 1) * lda] = d21 * (d22 * W[i + (k + 1) * ldw] - W[i + k * ldw]);
                    }
                }

                if(tid == 0)
                {
                    A[k + k * lda] = W[k + k * ldw];
                    A[(k + 1) + k * lda] = W[(k + 1) + k * ldw];
                    A[(k + 1) + (k + 1) * lda] = W[(k + 1) + (k + 1) * ldw];
                }
                __syncthreads();
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
    {
        *kb = k;
        *info = _info;
    }

    // update A from [k,k] to [n-1,n-1], nb columns at a time
    for(j = k; j < n; j += nb)
    {
        int jb = std::min(nb, n - j);
        for(i = j; i < j + jb; i++)
            lasyf_gemv<MAX_THDS>(tid, j + jb - i, k, minone, A + i, lda, W + i, ldw, one,
                                 A + i + i * lda, 1);
        if(j + jb < n)
            lasyf_gemm<MAX_THDS>(tid, n - j - jb, jb, k, minone, A + (j + jb), lda, W + j, ldw, one,
                                 A + (j + jb) + j * lda, lda);
    }
    __syncthreads();

    // partially undo interchanges to put L21 in standard form
    j = k - 1;
    while(j > 0)
    {
        kk = j; // jj
        kp = ipiv[j]; // jp
        if(kp < 0)
        {
            kp = -kp - 1;
            j--;
        }
        else
            kp = kp - 1;

        j--;
        if(kp != kk && j >= 0)
        {
            for(i = tid; i <= j; i += MAX_THDS)
                swap(A[kp + i * lda], A[kk + i * lda]);
            __syncthreads();
        }
    }
}

template <typename T, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(LASYF_MAX_THDS)
    lasyf_kernel_upper(const rocblas_int n,
                       const rocblas_int nb,
                       rocblas_int* kbA,
                       U AA,
                       const rocblas_int shiftA,
                       const rocblas_int lda,
                       const rocblas_stride strideA,
                       rocblas_int* ipivA,
                       const rocblas_stride strideP,
                       rocblas_int* infoA,
                       T* WA)
{
    using S = decltype(std::real(T{}));

    // select batch instance
    rocblas_int bid = hipBlockIdx_y;
    rocblas_int tid = hipThreadIdx_x;

    // get array pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* W = WA + (bid * n * nb);
    rocblas_int* ipiv = ipivA + (bid * strideP);

    // shared arrays
    __shared__ S sval[LASYF_MAX_THDS];
    __shared__ rocblas_int sidx[LASYF_MAX_THDS];

    lasyf_device_upper<LASYF_MAX_THDS>(tid, n, nb, kbA + bid, A, lda, ipiv, infoA + bid, W, sidx,
                                       sval);
}

template <typename T, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(LASYF_MAX_THDS)
    lasyf_kernel_lower(const rocblas_int n,
                       const rocblas_int nb,
                       rocblas_int* kbA,
                       U AA,
                       const rocblas_int shiftA,
                       const rocblas_int lda,
                       const rocblas_stride strideA,
                       rocblas_int* ipivA,
                       const rocblas_stride strideP,
                       rocblas_int* infoA,
                       T* WA)
{
    using S = decltype(std::real(T{}));

    // select batch instance
    rocblas_int bid = hipBlockIdx_y;
    rocblas_int tid = hipThreadIdx_x;

    // get array pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* W = WA + (bid * n * nb);
    rocblas_int* ipiv = ipivA + (bid * strideP);

    // shared arrays
    __shared__ S sval[LASYF_MAX_THDS];
    __shared__ rocblas_int sidx[LASYF_MAX_THDS];

    lasyf_device_lower<LASYF_MAX_THDS>(tid, n, nb, kbA + bid, A, lda, ipiv, infoA + bid, W, sidx,
                                       sval);
}

template <typename T>
void rocsolver_lasyf_getMemorySize(const rocblas_int n,
                                   const rocblas_int nb,
                                   const rocblas_int batch_count,
                                   size_t* size_work)
{
    // if quick return no workspace needed
    if(n == 0 || nb == 0 || batch_count == 0)
    {
        *size_work = 0;
        return;
    }

    // size of workspace
    *size_work = sizeof(T) * n * nb * batch_count;
}

template <typename T, typename U>
rocblas_status rocsolver_lasyf_argCheck(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        const rocblas_int nb,
                                        const rocblas_int lda,
                                        U kb,
                                        T A,
                                        U ipiv,
                                        U info,
                                        const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || nb < 0 || nb > n || lda < n || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((batch_count && !kb) || (n && !A) || (n && !ipiv) || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U>
rocblas_status rocsolver_lasyf_template(rocblas_handle handle,
                                        const rocblas_fill uplo,
                                        const rocblas_int n,
                                        const rocblas_int nb,
                                        rocblas_int* kb,
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
    ROCSOLVER_ENTER("lasyf", "uplo:", uplo, "n:", n, "nb:", nb, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // quick return
    if(n == 0 || nb == 0)
    {
        // set info = 0
        rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
        dim3 gridReset(blocksReset, 1, 1);
        dim3 threadsReset(BS1, 1, 1);
        ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threadsReset, 0, stream, kb, batch_count, 0);
        ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threadsReset, 0, stream, info, batch_count, 0);

        return rocblas_status_success;
    }

    dim3 grid(1, batch_count, 1);
    dim3 threads(LASYF_MAX_THDS, 1, 1);

    if(uplo == rocblas_fill_upper)
        ROCSOLVER_LAUNCH_KERNEL(lasyf_kernel_upper<T>, grid, threads, 0, stream, n, nb, kb, A,
                                shiftA, lda, strideA, ipiv, strideP, info, work);
    else
        ROCSOLVER_LAUNCH_KERNEL(lasyf_kernel_lower<T>, grid, threads, 0, stream, n, nb, kb, A,
                                shiftA, lda, strideA, ipiv, strideP, info, work);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE

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
#include "rocsolver_prng.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/** thread-block size for calling the stein kernel.
    (MAX_THDS sizes must be one of 128, 256, 512, or 1024) **/
#define STEIN_MAX_THDS 256

#define STEIN_MAX_ITERS 5

#define STEIN_MAX_NRMCHK 2

template <typename T, typename S, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
__device__ void stein_reorthogonalize(rocblas_int i,
                                      const rocblas_int j,
                                      const rocblas_int n,
                                      const rocblas_int b1,
                                      S* work,
                                      T* Z,
                                      const rocblas_int ldz)
{
    S ztr;
    rocblas_int jr;

    for(; i <= j - 1; i++)
    {
        ztr = 0;
        for(jr = 0; jr < n; jr++)
            ztr = ztr + work[jr] * Z[(b1 + jr) + i * ldz];
        for(jr = 0; jr < n; jr++)
            work[jr] = work[jr] - ztr * Z[(b1 + jr) + i * ldz];
    }
}

template <typename T, typename S, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
__device__ void stein_reorthogonalize(rocblas_int i,
                                      const rocblas_int j,
                                      const rocblas_int n,
                                      const rocblas_int b1,
                                      S* work,
                                      T* Z,
                                      const rocblas_int ldz)
{
    S ztr;
    rocblas_int jr;

    for(; i <= j - 1; i++)
    {
        ztr = 0;
        for(jr = 0; jr < n; jr++)
            ztr = ztr + work[jr] * Z[(b1 + jr) + i * ldz].real();
        for(jr = 0; jr < n; jr++)
            work[jr] = work[jr] - ztr * Z[(b1 + jr) + i * ldz].real();
    }
}

template <int MAX_THDS, typename T, typename S>
__device__ void run_stein(const int tid,
                          const rocblas_int n,
                          S* D,
                          S* E,
                          const rocblas_int nev,
                          S* W,
                          rocblas_int* iblock,
                          rocblas_int* isplit,
                          T* Z,
                          const rocblas_int ldz,
                          rocblas_int* ifail,
                          rocblas_int* info,
                          S* work,
                          rocblas_int* iwork,
                          S* sval1,
                          S* sval2,
                          rocblas_int* sidx,
                          S eps,
                          S ssfmin)
{
    __shared__ rocblas_int _info;
    rocblas_int i, j, j1 = 0, b1, bn, blksize, gpind, seed;
    S scl, onenrm, ortol, stpcrt, xj, xjm;

    // zero info and ifail
    if(tid == 0)
        _info = 0;
    if(ifail)
        for(i = tid; i < nev; i += MAX_THDS)
            ifail[i] = 0;

    auto normalize = [](rocblas_int value, rocblas_int max) -> S {
        return static_cast<S>(static_cast<double>(value) / max);
    };
    // iterate over submatrix blocks
    for(rocblas_int nblk = 0; nblk < iblock[nev - 1]; nblk++)
    {
        // start and end indices of the submatrix
        b1 = (nblk == 0 ? 0 : isplit[nblk - 1]);
        bn = isplit[nblk] - 1;
        blksize = bn - b1 + 1;

        if(blksize > 1)
        {
            gpind = j1;

            // compute reorthogonalization criterion and stopping criterion
            onenrm = abs(D[b1]) + abs(E[b1]);
            onenrm = std::max(onenrm, abs(D[bn]) + abs(E[bn - 1]));
            for(j = b1 + 1; j <= bn - 1; j++)
                onenrm = std::max(onenrm, abs(D[j]) + abs(E[j - 1]) + abs(E[j])); // <- parallelize?
            ortol = S(0.001) * onenrm;
            stpcrt = sqrt(0.1 / blksize);
        }

        // loop through eigenvalues for current block
        rocblas_int jblk = 0;
        for(j = j1; j < nev; j++)
        {
            if(iblock[j] - 1 != nblk)
            {
                j1 = j;
                break;
            }

            jblk++;
            xj = W[j];

            if(blksize > 1)
            {
                // if eigenvalues j and j-1 are too close, add a perturbation
                if(jblk > 1)
                {
                    S pertol = 10 * abs(eps * xj);
                    if(xj - xjm < pertol)
                        xj = xjm + pertol;
                }

                rocblas_int iters = 0;
                rocblas_int nrmchk = 0;

                // initialize starting eigenvector

                // deterministic seed, varies with thread ids and with each new computed eigenvalue
                seed = tid + (j + nblk * nev) * MAX_THDS + 1;

                // rocSOLVER's prng can be initialized with one or two seeds (better results are achieved if
                // two, independent, seeds are used).  The two seeds used here are not independent, they just spread
                // the first pseudo-random number each thread computes more evenly.
                rocsolver_int_prng<rocblas_int> prng(
                    seed, rocsolver_int_prng<rocblas_int>::max() + 1 - seed);

                for(i = tid; i < blksize; i += MAX_THDS)
                {
                    // work[i] lies on the interval [0, 1]
                    work[i] = normalize(prng(), prng.max());
                }

                // copy the matrix so it won't be destroyed by factorization
                for(i = tid; i < blksize - 1; i += MAX_THDS)
                {
                    work[3 * n + i] = D[b1 + i];
                    work[2 * n + i] = E[b1 + i];
                    work[n + i + 1] = E[b1 + i];
                }
                if(tid == 0)
                    work[3 * n + blksize - 1] = D[bn];
                __syncthreads();

                // compute LU factors with partial pivoting
                if(tid == 0)
                    lagtf<S>(blksize, work + 3 * n, xj, work + n + 1, work + 2 * n, 0, work + 4 * n,
                             iwork, eps);

                while(iters < STEIN_MAX_ITERS && nrmchk < STEIN_MAX_NRMCHK)
                {
                    // normalize and scale righthand side vector
                    iamax<MAX_THDS, S>(tid, blksize, work, 1, sval1, sidx);
                    __syncthreads();
                    scl = blksize * onenrm * std::max(eps, abs(work[3 * n + blksize - 1])) / sval1[0];
                    for(i = tid; i < blksize; i += MAX_THDS) // <- scal
                        work[i] = work[i] * scl;
                    __syncthreads();

                    // solve the system
                    if(tid == 0)
                        lagts_type1_perturb<S>(blksize, work + 3 * n, work + n + 1, work + 2 * n,
                                               work + 4 * n, iwork, work, 0, eps, ssfmin);
                    __syncthreads();

                    // reorthogonalize by modified Gram-Schmidt if eigenvalues are close enough
                    if(jblk > 1)
                    {
                        if(abs(xj - xjm) > ortol)
                            gpind = j;
                        if(gpind != j)
                        {
                            if(tid == 0)
                                stein_reorthogonalize<T>(gpind, j, blksize, b1, work, Z, ldz);
                            __syncthreads();
                        }
                    }

                    // check the infinity norm of the iterate against stopping condition
                    iamax<MAX_THDS, S>(tid, blksize, work, 1, sval1, sidx);
                    __syncthreads();
                    if(sval1[0] >= stpcrt)
                        nrmchk++;

                    iters++;
                }

                if(ifail && tid == 0 && nrmchk < STEIN_MAX_NRMCHK)
                {
                    ifail[_info] = j + 1;
                    _info++;
                }

                iamax<MAX_THDS, S>(tid, blksize, work, 1, sval1, sidx);
                nrm2<MAX_THDS, S>(tid, blksize, work, 1, sval2);
                __syncthreads();
                scl = (work[sidx[0] - 1] >= 0 ? S(1) / sval2[0] : S(-1) / sval2[0]);
                __syncthreads();
                for(i = tid; i < blksize; i += MAX_THDS) // <- scal
                    work[i] = work[i] * scl;
                __syncthreads();
            }
            else
            {
                if(tid == 0)
                    work[0] = S(1);
                __syncthreads();
            }

            for(i = tid; i < n; i += MAX_THDS)
                Z[i + j * ldz] = (i >= b1 && i <= bn ? work[i - b1] : T(0));
            __syncthreads();

            xjm = xj;
        }
    }

    if(tid == 0)
        *info = _info;
}

template <typename T, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(STEIN_MAX_THDS)
    stein_kernel(const rocblas_int n,
                 S* D,
                 const rocblas_stride strideD,
                 S* E,
                 const rocblas_stride strideE,
                 rocblas_int* nev,
                 S* W,
                 const rocblas_stride strideW,
                 rocblas_int* iblock,
                 const rocblas_stride strideIblock,
                 rocblas_int* isplit,
                 const rocblas_stride strideIsplit,
                 U ZZ,
                 const rocblas_int shiftZ,
                 const rocblas_int ldz,
                 const rocblas_stride strideZ,
                 rocblas_int* ifailA,
                 const rocblas_stride strideIfail,
                 rocblas_int* info,
                 S* work,
                 rocblas_int* iwork,
                 S eps,
                 S ssfmin)
{
    // select batch instance
    rocblas_int bid = hipBlockIdx_y;
    rocblas_int tid = hipThreadIdx_x;
    rocblas_stride stride_work = 5 * n;
    rocblas_stride stride_iwork = n;

    if(nev[bid] <= 0)
        return;

    T* Z = load_ptr_batch<T>(ZZ, bid, shiftZ, strideZ);
    rocblas_int* ifail = nullptr;
    if(ifailA)
        ifail = ifailA + (bid * strideIfail);

    // shared mem for temporary values
    extern __shared__ double lmem[];
    S* sval1 = reinterpret_cast<S*>(lmem);
    S* sval2 = reinterpret_cast<S*>(sval1 + STEIN_MAX_THDS);
    rocblas_int* sidx = reinterpret_cast<rocblas_int*>(sval2 + STEIN_MAX_THDS);

    // execute
    run_stein<STEIN_MAX_THDS, T>(
        tid, n, D + (bid * strideD), E + (bid * strideE), nev[bid], W + (bid * strideW),
        iblock + (bid * strideIblock), isplit + (bid * strideIsplit), Z, ldz, ifail, info + bid,
        work + (bid * stride_work), iwork + (bid * stride_iwork), sval1, sval2, sidx, eps, ssfmin);
}

template <typename T, typename S>
void rocsolver_stein_getMemorySize(const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_work,
                                   size_t* size_iwork)
{
    // if quick return no workspace needed
    if(n == 0 || !batch_count)
    {
        *size_work = 0;
        *size_iwork = 0;
        return;
    }

    // size of workspace
    *size_work = sizeof(S) * 5 * n * batch_count;

    // size of integer workspace
    *size_iwork = sizeof(rocblas_int) * n * batch_count;
}

template <typename T, typename S>
rocblas_status rocsolver_stein_argCheck(rocblas_handle handle,
                                        const rocblas_int n,
                                        S* D,
                                        S* E,
                                        rocblas_int* nev,
                                        S* W,
                                        rocblas_int* iblock,
                                        rocblas_int* isplit,
                                        T* Z,
                                        const rocblas_int ldz,
                                        rocblas_int* ifail,
                                        rocblas_int* info)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(n < 0 || ldz < n)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !D) || (n && !E) || !nev || (n && !W) || (n && !iblock) || (n && !isplit) || (n && !Z)
       || (n && !ifail) || !info)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename S, typename U>
rocblas_status rocsolver_stein_template(rocblas_handle handle,
                                        const rocblas_int n,
                                        S* D,
                                        const rocblas_int shiftD,
                                        const rocblas_stride strideD,
                                        S* E,
                                        const rocblas_int shiftE,
                                        const rocblas_stride strideE,
                                        rocblas_int* nev,
                                        S* W,
                                        const rocblas_int shiftW,
                                        const rocblas_stride strideW,
                                        rocblas_int* iblock,
                                        const rocblas_stride strideIblock,
                                        rocblas_int* isplit,
                                        const rocblas_stride strideIsplit,
                                        U Z,
                                        const rocblas_int shiftZ,
                                        const rocblas_int ldz,
                                        const rocblas_stride strideZ,
                                        rocblas_int* ifail,
                                        const rocblas_stride strideIfail,
                                        rocblas_int* info,
                                        const rocblas_int batch_count,
                                        S* work,
                                        rocblas_int* iwork)
{
    ROCSOLVER_ENTER("stein", "n:", n, "shiftD:", shiftD, "shiftE:", shiftE, "shiftW:", shiftW,
                    "shiftZ:", shiftZ, "ldz:", ldz, "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threadsReset(BS1, 1, 1);

    // info = 0
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threadsReset, 0, stream, info, batch_count, 0);

    // quick return
    if(n == 0)
        return rocblas_status_success;

    S eps = get_epsilon<T>();
    S ssfmin = get_safemin<T>();

    dim3 grid(1, batch_count, 1);
    dim3 threads(STEIN_MAX_THDS, 1, 1);
    size_t lmemsize = STEIN_MAX_THDS * (2 * sizeof(S) + sizeof(rocblas_int));
    ROCSOLVER_LAUNCH_KERNEL(stein_kernel<T>, grid, threads, lmemsize, stream, n, D + shiftD,
                            strideD, E + shiftE, strideE, nev, W + shiftW, strideW, iblock,
                            strideIblock, isplit, strideIsplit, Z, shiftZ, ldz, strideZ, ifail,
                            strideIfail, info, work, iwork, eps, ssfmin);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE

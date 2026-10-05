/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.12.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2023
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "auxiliary/rocauxiliary_lahqr.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

/** TREXC_BLOCK reorders the Schur factorization of a complex matrix, A = Q*T*Q^H,
    so that the diagonal entry of the upper triangular T at row ifst is moved to
    row ilst, following LAPACK ZTREXC (a sequence of swaps of adjacent diagonal
    entries by plane rotations).

    On the device it must be called by all the threads of a thread-block of BS threads;
    on the host (the hybrid mode of HSEQR), with BS = 1. The indices
    ifst and ilst are 1-based, as in LAPACK, and must satisfy 1 <= ifst, ilst <= n.
    The rotations of each swap are computed redundantly by all the threads, and their
    application to the rows and columns of T and Q is distributed among the threads. **/
template <int BS, typename T, typename I>
__host__ __device__ __forceinline__ void trexc_block(const bool wantq,
                                                     const I n,
                                                     T* A,
                                                     const I ldt,
                                                     T* Q,
                                                     const I ldq,
                                                     const I ifst,
                                                     const I ilst)
{
    using S = decltype(std::real(T{}));

    const I tid = hqr_tid();
    auto t = [&](const I i, const I j) -> T& { return A[idx2D(i - 1, j - 1, ldt)]; };
    auto q = [&](const I i, const I j) -> T& { return Q[idx2D(i - 1, j - 1, ldq)]; };

    // quick return if possible
    if(n <= 1 || ifst == ilst)
        return;

    // move the ifst-th diagonal element forward down the diagonal, or backward up
    // the diagonal
    const I m1 = (ifst < ilst) ? 0 : -1;
    const I m2 = (ifst < ilst) ? -1 : 0;
    const I m3 = (ifst < ilst) ? 1 : -1;
    const I kfirst = ifst + m1;
    const I klast = ilst + m2;
    const I nswaps = (klast - kfirst) / m3 + 1;

    for(I sw = 0; sw < nswaps; sw++)
    {
        const I k = kfirst + sw * m3;

        // interchange the k-th and (k+1)-th diagonal elements
        const T t11 = t(k, k);
        const T t22 = t(k + 1, k + 1);

        // determine the transformation to perform the interchange
        S cs;
        T sn, temp;
        hqr_lartg(t(k, k + 1), t22 - t11, cs, sn, temp);

        // apply the transformation to the matrix T (T(k,k+1) does not change)
        for(I j = k + 2 + tid; j <= n; j += BS)
            hqr_rot(t(k, j), t(k + 1, j), cs, sn);
        for(I j = 1 + tid; j <= k - 1; j += BS)
            hqr_rot(t(j, k), t(j, k + 1), cs, conj(sn));

        // accumulate the transformation in the matrix Q
        if(wantq)
            for(I j = 1 + tid; j <= n; j += BS)
                hqr_rot(q(j, k), q(j, k + 1), cs, conj(sn));

        // all the threads have read t11, t22 and t(k,k+1)
        hqr_sync();
        if(tid == 0)
        {
            t(k, k) = t22;
            t(k + 1, k + 1) = t11;
        }
        hqr_sync();
    }
}

template <int BS, typename T, typename I, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(BS) trexc_kernel(const rocsolver_schur_vectors compq,
                                                         const I n,
                                                         U AA,
                                                         const rocblas_stride shiftT,
                                                         const I ldt,
                                                         const rocblas_stride strideT,
                                                         U QQ,
                                                         const rocblas_stride shiftQ,
                                                         const I ldq,
                                                         const rocblas_stride strideQ,
                                                         const I ifst,
                                                         const I ilst)
{
    const I bid = hipBlockIdx_x;
    const bool wantq = (compq == rocsolver_schur_vectors_update);

    T* A = load_ptr_batch<T>(AA, bid, shiftT, strideT);
    T* Q = wantq ? load_ptr_batch<T>(QQ, bid, shiftQ, strideQ) : nullptr;

    trexc_block<BS>(wantq, n, A, ldt, Q, ldq, ifst, ilst);
}

template <typename I, typename U>
rocblas_status rocsolver_trexc_argCheck(rocblas_handle handle,
                                        const rocsolver_schur_vectors compq,
                                        const I n,
                                        const I ldt,
                                        const I ldq,
                                        const I ifst,
                                        const I ilst,
                                        U A,
                                        U Q,
                                        const I batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(compq != rocsolver_schur_vectors_none && compq != rocsolver_schur_vectors_update)
        return rocblas_status_invalid_value;
    const bool wantq = (compq == rocsolver_schur_vectors_update);

    // 2. invalid size
    if(n < 0 || ldt < n || ldt < 1 || ldq < 1 || (wantq && ldq < n) || batch_count < 0)
        return rocblas_status_invalid_size;
    if(n > 0 && (ifst < 1 || ifst > n || ilst < 1 || ilst > n))
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && wantq && !Q))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename U>
rocblas_status rocsolver_trexc_template(rocblas_handle handle,
                                        const rocsolver_schur_vectors compq,
                                        const I n,
                                        U A,
                                        const rocblas_stride shiftT,
                                        const I ldt,
                                        const rocblas_stride strideT,
                                        U Q,
                                        const rocblas_stride shiftQ,
                                        const I ldq,
                                        const rocblas_stride strideQ,
                                        const I ifst,
                                        const I ilst,
                                        const I batch_count)
{
    ROCSOLVER_ENTER("trexc", "compq:", compq, "n:", n, "shiftT:", shiftT, "ldt:", ldt,
                    "shiftQ:", shiftQ, "ldq:", ldq, "ifst:", ifst, "ilst:", ilst, "bc:", batch_count);

    // quick return
    if(n <= 1 || ifst == ilst || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    ROCSOLVER_LAUNCH_KERNEL((trexc_kernel<TREXC_BLOCKSIZE, T>), dim3(batch_count),
                            dim3(TREXC_BLOCKSIZE), 0, stream, compq, n, A, shiftT, ldt, strideT, Q,
                            shiftQ, ldq, strideQ, ifst, ilst);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE

/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routines (version 3.12.0) --
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

#include "rocauxiliary_geequ.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/** Scale factors of a positive definite matrix from its diagonal, as in xPOEQU (or xPOEQUB if
    POW2). Grid (1, 1, batch_count), block EQU_FINAL_THDS. **/
template <bool POW2, typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(EQU_FINAL_THDS) poequ_kernel(const I n,
                                                                     U A,
                                                                     const rocblas_stride shiftA,
                                                                     const I lda,
                                                                     const rocblas_stride strideA,
                                                                     S* Sc,
                                                                     const rocblas_stride strideS,
                                                                     S* scond,
                                                                     S* amax,
                                                                     I* info)
{
    const I b = blockIdx.z;
    const I tid = threadIdx.x;

    T* a = load_ptr_batch<T>(A, b, shiftA, strideA);
    S* s = Sc + b * strideS;

    __shared__ S smin[EQU_FINAL_THDS], smx[EQU_FINAL_THDS];
    __shared__ int64_t sbad[EQU_FINAL_THDS];

    // diagonal (real part), its extremes and the first nonpositive element
    S vmin = std::numeric_limits<S>::max(), vmax = std::numeric_limits<S>::lowest();
    int64_t bad = n;
    for(int64_t i = tid; i < n; i += EQU_FINAL_THDS)
    {
        const S v = std::real(a[idx2D(i, i, lda)]);
        s[i] = v;
        vmin = std::min(vmin, v);
        vmax = std::max(vmax, v);
        if(v <= 0)
            bad = std::min(bad, i);
    }

    smin[tid] = vmin;
    smx[tid] = vmax;
    sbad[tid] = bad;
    __syncthreads();
    for(I k = EQU_FINAL_THDS / 2; k > 0; k /= 2)
    {
        if(tid < k)
        {
            smin[tid] = std::min(smin[tid], smin[tid + k]);
            smx[tid] = std::max(smx[tid], smx[tid + k]);
            sbad[tid] = std::min(sbad[tid], sbad[tid + k]);
        }
        __syncthreads();
    }
    vmin = smin[0];
    vmax = smx[0];
    bad = sbad[0];

    if(tid == 0)
    {
        amax[b] = vmax;
        info[b] = (bad < n) ? bad + 1 : 0;
    }

    // a nonpositive diagonal element: the diagonal is left in S, as in LAPACK
    if(bad < n)
        return;

    for(int64_t i = tid; i < n; i += EQU_FINAL_THDS)
    {
        if(POW2)
            s[i] = equ_pow2<true>(s[i]);
        else
            s[i] = S(1) / sqrt(s[i]);
    }

    if(tid == 0)
        scond[b] = sqrt(vmin) / sqrt(vmax);
}

template <typename T, typename I, typename S>
rocblas_status rocsolver_poequ_argCheck(rocblas_handle handle,
                                        const I n,
                                        const I lda,
                                        T A,
                                        S* Sc,
                                        S* scond,
                                        S* amax,
                                        I* info)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(n < 0 || lda < n)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && (!A || !Sc || !scond || !amax)) || !info)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

/** POEQU (POW2 = false) and POEQUB (POW2 = true). No workspace is needed. **/
template <bool POW2, typename T, typename I, typename S, typename U>
rocblas_status rocsolver_poequ_template(rocblas_handle handle,
                                        const I n,
                                        U A,
                                        const rocblas_stride shiftA,
                                        const I lda,
                                        const rocblas_stride strideA,
                                        S* Sc,
                                        const rocblas_stride strideS,
                                        S* scond,
                                        S* amax,
                                        I* info,
                                        const I batch_count)
{
    ROCSOLVER_ENTER((POW2 ? "poequb" : "poequ"), "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // quick return: scond = 1, amax = 0
    if(n == 0)
    {
        const I blocks = (batch_count - 1) / BS1 + 1;
        ROCSOLVER_LAUNCH_KERNEL((reset_info<I, I, I>), dim3(blocks), dim3(BS1), 0, stream, info,
                                batch_count, I(0));
        if(scond)
            ROCSOLVER_LAUNCH_KERNEL((reset_info<S, I, S>), dim3(blocks), dim3(BS1), 0, stream,
                                    scond, batch_count, S(1));
        if(amax)
            ROCSOLVER_LAUNCH_KERNEL((reset_info<S, I, S>), dim3(blocks), dim3(BS1), 0, stream, amax,
                                    batch_count, S(0));
        return rocblas_status_success;
    }

    ROCSOLVER_LAUNCH_KERNEL((poequ_kernel<POW2, T>), dim3(1, 1, batch_count), dim3(EQU_FINAL_THDS),
                            0, stream, n, A, shiftA, lda, strideA, Sc, strideS, scond, amax, info);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE

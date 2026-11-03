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

#include "ideal_sizes.hpp"
#include "lib_device_helpers.hpp"
#include "rocblas.hpp"
#include "rocblas_utility.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/*
 * LAQGE, LAQSY and LAQHE apply the scale factors computed by GEEQU or POEQU. The decision whether to
 * scale (as in LAPACK) depends on scalars on the device, so each thread makes it; if no scaling is
 * needed, the kernel returns without reading the matrix.
 */

#ifndef LAQ_BX
#define LAQ_BX 64
#endif
#ifndef LAQ_BY
#define LAQ_BY 4
#endif

/** Thresholds of xLAQGE, xLAQSY and xLAQHE: thresh = 0.1, and small = safe minimum / precision. **/
template <typename S>
__device__ __forceinline__ bool laq_amax_ok(const S amax)
{
    const S small = std::numeric_limits<S>::min() / std::numeric_limits<S>::epsilon();
    const S large = S(1) / small;
    return amax >= small && amax <= large;
}

/** Scales A_b by rows and/or columns, as decided by xLAQGE. Grid (ceil(m / LAQ_BX),
    ceil(n / (LAQ_BY * cols)), batch_count), block (LAQ_BX, LAQ_BY). **/
template <typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(LAQ_BX* LAQ_BY) laqge_kernel(const I m,
                                                                     const I n,
                                                                     U A,
                                                                     const rocblas_stride shiftA,
                                                                     const I lda,
                                                                     const rocblas_stride strideA,
                                                                     const S* R,
                                                                     const rocblas_stride strideR,
                                                                     const S* C,
                                                                     const rocblas_stride strideC,
                                                                     const S* rowcnd,
                                                                     const S* colcnd,
                                                                     const S* amax,
                                                                     rocsolver_equilibration* equed)
{
    const I b = blockIdx.z;
    const S thresh = S(0.1);

    rocsolver_equilibration eq;
    if(rowcnd[b] >= thresh && laq_amax_ok(amax[b]))
        eq = (colcnd[b] >= thresh) ? rocsolver_equilibration_none : rocsolver_equilibration_column;
    else if(colcnd[b] >= thresh)
        eq = rocsolver_equilibration_row;
    else
        eq = rocsolver_equilibration_both;

    if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
        equed[b] = eq;
    if(eq == rocsolver_equilibration_none)
        return;

    T* a = load_ptr_batch<T>(A, b, shiftA, strideA);
    const S* r = R + b * strideR;
    const S* c = C + b * strideC;
    const bool rows = (eq != rocsolver_equilibration_column);
    const bool cols = (eq != rocsolver_equilibration_row);

    for(int64_t j = blockIdx.y * int64_t(LAQ_BY) + threadIdx.y; j < n;
        j += int64_t(gridDim.y) * LAQ_BY)
    {
        const S cj = cols ? c[j] : S(1);
        for(int64_t i = blockIdx.x * int64_t(LAQ_BX) + threadIdx.x; i < m;
            i += int64_t(gridDim.x) * LAQ_BX)
        {
            // the products are formed in the order of LAPACK
            S f;
            if(rows && cols)
                f = cj * r[i];
            else if(rows)
                f = r[i];
            else
                f = cj;
            a[idx2D(i, j, lda)] = f * a[idx2D(i, j, lda)];
        }
    }
}

/** Scales the stored triangle of A_b symmetrically, as decided by xLAQSY (or xLAQHE if HERM). Grid
    and block as laqge_kernel. **/
template <bool HERM, typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(LAQ_BX* LAQ_BY) laqsy_kernel(const bool upper,
                                                                     const I n,
                                                                     U A,
                                                                     const rocblas_stride shiftA,
                                                                     const I lda,
                                                                     const rocblas_stride strideA,
                                                                     const S* Sc,
                                                                     const rocblas_stride strideS,
                                                                     const S* scond,
                                                                     const S* amax,
                                                                     rocsolver_equilibration* equed)
{
    const I b = blockIdx.z;
    const S thresh = S(0.1);

    const bool scale = !(scond[b] >= thresh && laq_amax_ok(amax[b]));

    if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
        equed[b] = scale ? rocsolver_equilibration_both : rocsolver_equilibration_none;
    if(!scale)
        return;

    T* a = load_ptr_batch<T>(A, b, shiftA, strideA);
    const S* s = Sc + b * strideS;

    for(int64_t j = blockIdx.y * int64_t(LAQ_BY) + threadIdx.y; j < n;
        j += int64_t(gridDim.y) * LAQ_BY)
    {
        const S cj = s[j];
        for(int64_t i = blockIdx.x * int64_t(LAQ_BX) + threadIdx.x; i < n;
            i += int64_t(gridDim.x) * LAQ_BX)
        {
            if(upper ? (i > j) : (i < j))
                continue;
            if(HERM && i == j)
                a[idx2D(i, j, lda)] = T(cj * cj * std::real(a[idx2D(i, j, lda)]));
            else
                a[idx2D(i, j, lda)] = (cj * s[i]) * a[idx2D(i, j, lda)];
        }
    }
}

template <typename T, typename I, typename S>
rocblas_status rocsolver_laqge_argCheck(rocblas_handle handle,
                                        const I m,
                                        const I n,
                                        const I lda,
                                        T A,
                                        const S* R,
                                        const S* C,
                                        const S* rowcnd,
                                        const S* colcnd,
                                        const S* amax,
                                        rocsolver_equilibration* equed)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    // N/A

    // 2. invalid size
    if(m < 0 || n < 0 || lda < m)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((m && n && (!A || !R || !C || !rowcnd || !colcnd || !amax)) || !equed)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename I, typename S>
rocblas_status rocsolver_laqsy_laqhe_argCheck(rocblas_handle handle,
                                              const rocblas_fill uplo,
                                              const I n,
                                              const I lda,
                                              T A,
                                              const S* Sc,
                                              const S* scond,
                                              const S* amax,
                                              rocsolver_equilibration* equed)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < n)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && (!A || !Sc || !scond || !amax)) || !equed)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

/** Sets equed = none for a batch (quick return). **/
template <typename I>
ROCSOLVER_KERNEL void laq_set_none(rocsolver_equilibration* equed, const I batch_count)
{
    const I b = blockIdx.x * I(blockDim.x) + threadIdx.x;
    if(b < batch_count)
        equed[b] = rocsolver_equilibration_none;
}

/** Grid of the scaling kernels: enough blocks to fill the device, looping over the rest. **/
template <typename I>
dim3 laq_grid(const I m, const I n, const I batch_count)
{
    const int64_t gx = std::min<int64_t>((int64_t(m) - 1) / LAQ_BX + 1, 1024);
    const int64_t gy = std::min<int64_t>((int64_t(n) - 1) / LAQ_BY + 1, 4096 / gx + 1);
    return dim3(gx, std::min<int64_t>(gy, 65535), batch_count);
}

template <typename T, typename I, typename S, typename U>
rocblas_status rocsolver_laqge_template(rocblas_handle handle,
                                        const I m,
                                        const I n,
                                        U A,
                                        const rocblas_stride shiftA,
                                        const I lda,
                                        const rocblas_stride strideA,
                                        const S* R,
                                        const rocblas_stride strideR,
                                        const S* C,
                                        const rocblas_stride strideC,
                                        const S* rowcnd,
                                        const S* colcnd,
                                        const S* amax,
                                        rocsolver_equilibration* equed,
                                        const I batch_count)
{
    ROCSOLVER_ENTER("laqge", "m:", m, "n:", n, "shiftA:", shiftA, "lda:", lda, "bc:", batch_count);

    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // quick return: equed = none
    if(m == 0 || n == 0)
    {
        ROCSOLVER_LAUNCH_KERNEL((laq_set_none<I>), dim3((batch_count - 1) / BS1 + 1), dim3(BS1), 0,
                                stream, equed, batch_count);
        return rocblas_status_success;
    }

    ROCSOLVER_LAUNCH_KERNEL((laqge_kernel<T>), laq_grid(m, n, batch_count), dim3(LAQ_BX, LAQ_BY), 0,
                            stream, m, n, A, shiftA, lda, strideA, R, strideR, C, strideC, rowcnd,
                            colcnd, amax, equed);

    return rocblas_status_success;
}

template <bool HERM, typename T, typename I, typename S, typename U>
rocblas_status rocsolver_laqsy_laqhe_template(rocblas_handle handle,
                                              const rocblas_fill uplo,
                                              const I n,
                                              U A,
                                              const rocblas_stride shiftA,
                                              const I lda,
                                              const rocblas_stride strideA,
                                              const S* Sc,
                                              const rocblas_stride strideS,
                                              const S* scond,
                                              const S* amax,
                                              rocsolver_equilibration* equed,
                                              const I batch_count)
{
    ROCSOLVER_ENTER((HERM ? "laqhe" : "laqsy"), "uplo:", uplo, "n:", n, "shiftA:", shiftA,
                    "lda:", lda, "bc:", batch_count);

    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // quick return: equed = none
    if(n == 0)
    {
        ROCSOLVER_LAUNCH_KERNEL((laq_set_none<I>), dim3((batch_count - 1) / BS1 + 1), dim3(BS1), 0,
                                stream, equed, batch_count);
        return rocblas_status_success;
    }

    ROCSOLVER_LAUNCH_KERNEL((laqsy_kernel<HERM, T>), laq_grid(n, n, batch_count),
                            dim3(LAQ_BX, LAQ_BY), 0, stream, uplo == rocblas_fill_upper, n, A,
                            shiftA, lda, strideA, Sc, strideS, scond, amax, equed);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE

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

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

/** GEBAK_SCALE_KERNEL scales rows ilo-1:ihi-1 of V by scale (side = right)
    or by 1/scale (side = left). Rows are skipped when ilo == ihi, as in LAPACK.
    Call with a 2D grid over the rows and columns of V, and the batch in z. **/
template <typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void gebak_scale_kernel(const rocblas_side side,
                                         const I n,
                                         const I m,
                                         const I* iloA,
                                         const I* ihiA,
                                         const S* scaleA,
                                         const rocblas_stride strideS,
                                         U VV,
                                         const rocblas_stride shiftV,
                                         const I ldv,
                                         const rocblas_stride strideV)
{
    const I bid = hipBlockIdx_z;
    const I i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const I j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    const I ilo = iloA[bid];
    const I ihi = ihiA[bid];
    if(ilo == ihi || i < ilo - 1 || i >= ihi || i >= n || j >= m)
        return;

    const S* scale = scaleA + bid * strideS;
    T* V = load_ptr_batch<T>(VV, bid, shiftV, strideV);

    S s = (side == rocblas_side_right) ? scale[i] : S(1) / scale[i];
    V[idx2D(i, j, ldv)] *= s;
}

/** GEBAK_PERMUTE_KERNEL undoes the permutations recorded in scale, in the same
    order as LAPACK: rows ilo-2 down to 0, then rows ihi to n-1. The swaps must be
    applied sequentially, but the columns of V are independent; each thread
    processes one column. Call with a 1D grid over the columns of V, and the batch in y. **/
template <typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void gebak_permute_kernel(const I n,
                                           const I m,
                                           const I* iloA,
                                           const I* ihiA,
                                           const S* scaleA,
                                           const rocblas_stride strideS,
                                           U VV,
                                           const rocblas_stride shiftV,
                                           const I ldv,
                                           const rocblas_stride strideV)
{
    const I bid = hipBlockIdx_y;
    const I j = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(j >= m)
        return;

    const I ilo = iloA[bid];
    const I ihi = ihiA[bid];
    const S* scale = scaleA + bid * strideS;
    T* V = load_ptr_batch<T>(VV, bid, shiftV, strideV);

    auto swap_rows = [&](const I i) {
        const I p = static_cast<I>(scale[i]) - 1;
        if(p != i && p >= 0 && p < n)
            swap(V[idx2D(i, j, ldv)], V[idx2D(p, j, ldv)]);
    };

    for(I i = std::min(ilo - 1, n) - 1; i >= 0; i--)
        swap_rows(i);
    for(I i = std::max(ihi, I(0)); i < n; i++)
        swap_rows(i);
}

template <typename T, typename I, typename S>
rocblas_status rocsolver_gebak_argCheck(rocblas_handle handle,
                                        const rocsolver_balance job,
                                        const rocblas_side side,
                                        const I n,
                                        const I* ilo,
                                        const I* ihi,
                                        const S* scale,
                                        const I m,
                                        T V,
                                        const I ldv,
                                        const I batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(job != rocsolver_balance_none && job != rocsolver_balance_permute
       && job != rocsolver_balance_scale && job != rocsolver_balance_both)
        return rocblas_status_invalid_value;
    if(side != rocblas_side_left && side != rocblas_side_right)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || m < 0 || ldv < n || ldv < 1 || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && m && !V) || (n && !scale) || (n && batch_count && !ilo) || (n && batch_count && !ihi))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename S, typename U>
rocblas_status rocsolver_gebak_template(rocblas_handle handle,
                                        const rocsolver_balance job,
                                        const rocblas_side side,
                                        const I n,
                                        const I* ilo,
                                        const I* ihi,
                                        const S* scale,
                                        const rocblas_stride strideS,
                                        const I m,
                                        U V,
                                        const rocblas_stride shiftV,
                                        const I ldv,
                                        const rocblas_stride strideV,
                                        const I batch_count)
{
    ROCSOLVER_ENTER("gebak", "job:", job, "side:", side, "n:", n, "m:", m, "shiftV:", shiftV,
                    "ldv:", ldv, "bc:", batch_count);

    // quick return
    if(n == 0 || m == 0 || batch_count == 0 || job == rocsolver_balance_none)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // backward balance
    if(job == rocsolver_balance_scale || job == rocsolver_balance_both)
    {
        const I blocksx = (n - 1) / BS2 + 1;
        const I blocksy = (m - 1) / BS2 + 1;
        ROCSOLVER_LAUNCH_KERNEL((gebak_scale_kernel<T>), dim3(blocksx, blocksy, batch_count),
                                dim3(BS2, BS2), 0, stream, side, n, m, ilo, ihi, scale, strideS, V,
                                shiftV, ldv, strideV);
    }

    // backward permutation
    if(job == rocsolver_balance_permute || job == rocsolver_balance_both)
    {
        const I blocks = (m - 1) / BS1 + 1;
        ROCSOLVER_LAUNCH_KERNEL((gebak_permute_kernel<T>), dim3(blocks, batch_count), dim3(BS1), 0,
                                stream, n, m, ilo, ihi, scale, strideS, V, shiftV, ldv, strideV);
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE

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

// thread-block size of gebak_gather_kernel, number of rows per thread (so that the
// swaps are composed at once for n <= GEBAK_GATHER_THDS * GEBAK_GATHER_ROWS, and in
// segments of GEBAK_GATHER_THDS * GEBAK_GATHER_ROWS / 2 swaps for larger n), and number
// of columns of each matrix that it permutes at a time
#ifndef GEBAK_GATHER_THDS
#define GEBAK_GATHER_THDS 1024
#endif
#ifndef GEBAK_GATHER_ROWS
#define GEBAK_GATHER_ROWS 8
#endif
#ifndef GEBAK_GATHER_COLS
#define GEBAK_GATHER_COLS 64
#endif

/** GEBAK_SCALE_KERNEL scales rows ilo-1:ihi-1 of V by scale (side = right)
    or by 1/scale (side = left). Rows are skipped when ilo == ihi, as in LAPACK.
    Matrices with infoA > 0 (if infoA is not null) are skipped.
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
                                         const rocblas_stride strideV,
                                         const I* infoA)
{
    const I bid = hipBlockIdx_z;
    const I i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const I j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(infoA && infoA[bid] > 0)
        return;
    const I ilo = iloA[bid];
    const I ihi = ihiA[bid];
    if(ilo == ihi || i < ilo - 1 || i >= ihi || i >= n || j >= m)
        return;

    const S* scale = scaleA + bid * strideS;
    T* V = load_ptr_batch<T>(VV, bid, shiftV, strideV);

    S s = (side == rocblas_side_right) ? scale[i] : S(1) / scale[i];
    V[idx2D(i, j, ldv)] *= s;
}

/** GEBAK_GATHER_KERNEL undoes the permutations recorded in scale, in the same order
    as LAPACK: rows ilo-2 down to 0, then rows ihi to n-1 are swapped with rows
    scale(i)-1 (an invalid index means no swap). As V is permuted by the swaps s_1, ...,
    s_K in this order, the row r of the result is the row perm(r) = s_1(s_2(...s_K(r)))
    of V, so the swaps are composed by following the rows through them in reverse order
    (rows n-1 down to ihi, then rows 0 to ilo-2, staged in shared memory).
    If SEG is false, each thread (x, y) owns the rows r = x + k * blockDim.x, k < ROWS,
    and all the swaps are composed at once. If SEG is true (for any n), the swaps are
    instead applied in segments of up to L = blockDim.x * ROWS / 2 consecutive swaps,
    starting from s_K. A segment only moves the at most 2L rows i and scale(i)-1 of its
    swaps, so each thread owns the entries e = x + k * blockDim.x of the list of these
    rows (first the rows i, which are consecutive, for coalesced stores, then the rows
    scale(i)-1; a row that is repeated is owned by its first entry only).
    Then, for the columns j = blockIdx.x * blockDim.y + y + l * ncols, ncols = gridDim.x
    * blockDim.y, it loads the rows perm(r) that move, and, after a barrier, stores them
    as rows r (threads along the rows, for coalesced accesses).
    Matrices with infoA > 0 (if infoA is not null) are skipped.
    Call with blockDim.x * blockDim.y <= GEBAK_GATHER_THDS, n <= blockDim.x * ROWS
    if SEG is false, and the batch in y. **/
template <int ROWS, bool SEG, typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(GEBAK_GATHER_THDS)
    gebak_gather_kernel(const I n,
                        const I m,
                        const I* iloA,
                        const I* ihiA,
                        const S* scaleA,
                        const rocblas_stride strideS,
                        U VV,
                        const rocblas_stride shiftV,
                        const I ldv,
                        const rocblas_stride strideV,
                        const I* infoA)
{
    const I bid = hipBlockIdx_y;
    const I tx = hipThreadIdx_x;
    const I ty = hipThreadIdx_y;
    const I dimx = hipBlockDim_x;
    const I tid = tx + ty * dimx;
    const I nt = dimx * hipBlockDim_y;
    if(infoA && infoA[bid] > 0)
        return;

    // numbers of swaps of rows ihi:n-1 and of rows 0:ilo-2 (nothing to do without swaps,
    // as for most matrices)
    const I ilo = iloA[bid];
    const I ihi = ihiA[bid];
    const I nb = n - std::min(std::max(ihi, I(0)), n);
    const I na = std::max(std::min(ilo - 1, n), I(0));
    const I ns = nb + na;
    if(ns == 0)
        return;

    const S* scale = scaleA + bid * strideS;
    T* V = load_ptr_batch<T>(VV, bid, shiftV, strideV);
    __shared__ I si[GEBAK_GATHER_THDS];
    __shared__ I sp[GEBAK_GATHER_THDS];

    // the row of the t-th swap in reverse order, and the row it is swapped with
    auto swap_row = [&](const I t) { return (t < nb) ? n - 1 - t : t - nb; };
    auto swap_with = [&](const I i) {
        const I p = static_cast<I>(scale[i]) - 1;
        return (p >= 0 && p < n) ? p : i;
    };

    // the swaps t0:t1-1 of each segment, from the last segment to the first (all the
    // threads of the block go through the loops the same number of times)
    const I ncols = hipGridDim_x * hipBlockDim_y;
    const I nseg = SEG ? (dimx * ROWS) / 2 : ns;
    for(I t1 = ns; t1 > 0; t1 -= nseg)
    {
        const I t0 = std::max(t1 - nseg, I(0));
        const I len = t1 - t0;

        // with SEG, the rows i of the segment are n-1-t for t0 <= t < tb, and t-nb for
        // tb <= t < t1, so that a row can be compared with all of them at once
        const I tb = std::min(std::max(nb, t0), t1);
        auto in_rows_i = [&](const I x, const bool top) {
            return (x >= n - tb && x < n - t0) || (top && x >= tb - nb && x < t1 - nb);
        };

        // rows r owned by the thread (-1 if none with SEG; rows r >= n are never swapped, so
        // that q = r for them), and the rows q = perm(r). With SEG, the swap u = t - t0 of
        // the segment has the entries u and len + u, and an entry is dropped if a smaller one
        // has the same row: the rows i can only repeat if the ranges of rows 0:ilo-2 and
        // ihi:n-1 overlap, and the rows scale(i)-1 are first compared with the rows i
        I r[ROWS], q[ROWS];
#pragma unroll
        for(int k = 0; k < ROWS; k++)
        {
            const I e = tx + k * dimx;
            if constexpr(SEG)
            {
                if(e < len)
                {
                    r[k] = swap_row(t0 + e);
                    if(t0 + e >= tb && in_rows_i(r[k], false))
                        r[k] = -1;
                }
                else if(e < 2 * len)
                {
                    r[k] = swap_with(swap_row(t0 + e - len));
                    if(in_rows_i(r[k], true))
                        r[k] = -1;
                }
                else
                    r[k] = -1;
            }
            else
                r[k] = e;
            q[k] = r[k];
        }

        // compose the swaps (and, with SEG, drop the entries len + u whose row scale(i)-1
        // is also the row of a smaller entry len + u')
        for(I c0 = t0; c0 < t1; c0 += nt)
        {
            const I t = c0 + tid;
            if(t < t1)
            {
                const I i = swap_row(t);
                si[tid] = i;
                sp[tid] = swap_with(i);
            }
            __syncthreads();

            // (the swap l of the chunk has the entry len + c0 - t0 + l, which comes before the
            // entry tx + k * dimx if l < el + k * dimx; these are bounded by the number of
            // entries, so that 32 bits are enough and spare registers with 64-bit integers)
            const int tn = static_cast<int>(std::min(nt, t1 - c0));
            const int el = static_cast<int>(tx - len - (c0 - t0));
            for(int l = 0; l < tn; l++)
            {
                const I i = si[l];
                const I p = sp[l];
#pragma unroll
                for(int k = 0; k < ROWS; k++)
                {
                    q[k] = (q[k] == i) ? p : (q[k] == p ? i : q[k]);
                    if constexpr(SEG)
                    {
                        if(p == r[k] && l < el + k * static_cast<int>(dimx))
                            r[k] = -1;
                    }
                }
            }
            __syncthreads();
        }

        // permute the columns (the real and imaginary parts are kept apart, so that they
        // stay in registers; the barrier at the start of the next segment orders its loads
        // after these stores)
        S vr[ROWS], vi[ROWS];
        for(I j0 = hipBlockIdx_x * hipBlockDim_y; j0 < m; j0 += ncols)
        {
            const I j = j0 + ty;
#pragma unroll
            for(int k = 0; k < ROWS; k++)
            {
                if(j < m && q[k] != r[k] && (!SEG || r[k] >= 0))
                {
                    const T x = V[idx2D(q[k], j, ldv)];
                    vr[k] = std::real(x);
                    vi[k] = std::imag(x);
                }
            }
            __syncthreads();

#pragma unroll
            for(int k = 0; k < ROWS; k++)
            {
                if(j < m && q[k] != r[k] && (!SEG || r[k] >= 0))
                {
                    if constexpr(rocblas_is_complex<T>)
                        V[idx2D(r[k], j, ldv)] = T(vr[k], vi[k]);
                    else
                        V[idx2D(r[k], j, ldv)] = vr[k];
                }
            }
        }
    }
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

/** The optional info is internal (for GEEV): the matrices with info > 0 are left
    unchanged. **/
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
                                        const I batch_count,
                                        const I* info = nullptr)
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
                                shiftV, ldv, strideV, info);
    }

    // backward permutation
    if(job == rocsolver_balance_permute || job == rocsolver_balance_both)
    {
        // threads along the rows: the smallest power of 2 that is at least n, or at least 32
        // with n <= tx * rows, for rows = 4 or GEBAK_GATHER_ROWS (GEBAK_GATHER_THDS for
        // larger n, which then uses segments); the other threads of the block (up to
        // GEBAK_GATHER_THDS) along the columns
        I tx = 1, ty = 1;
        while(tx < GEBAK_GATHER_THDS && tx < n && (tx < 32 || tx * 4 < n))
            tx *= 2;
        while(tx * ty < GEBAK_GATHER_THDS && ty < m)
            ty *= 2;
        const I blocks = (std::min(m, I(GEBAK_GATHER_COLS)) - 1) / ty + 1;
        if(n <= tx * 4)
            ROCSOLVER_LAUNCH_KERNEL((gebak_gather_kernel<4, false, T>), dim3(blocks, batch_count),
                                    dim3(tx, ty), 0, stream, n, m, ilo, ihi, scale, strideS, V,
                                    shiftV, ldv, strideV, info);
        else if(n <= tx * GEBAK_GATHER_ROWS)
            ROCSOLVER_LAUNCH_KERNEL((gebak_gather_kernel<GEBAK_GATHER_ROWS, false, T>),
                                    dim3(blocks, batch_count), dim3(tx, ty), 0, stream, n, m, ilo,
                                    ihi, scale, strideS, V, shiftV, ldv, strideV, info);
        else
            ROCSOLVER_LAUNCH_KERNEL((gebak_gather_kernel<GEBAK_GATHER_ROWS, true, T>),
                                    dim3(blocks, batch_count), dim3(tx, ty), 0, stream, n, m, ilo,
                                    ihi, scale, strideS, V, shiftV, ldv, strideV, info);
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE

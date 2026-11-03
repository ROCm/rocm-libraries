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
#include "lib_host_helpers.hpp"
#include "rocblas.hpp"
#include "rocblas_utility.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/*
 * Equilibration of general (GEEQU, GEEQUB) and positive definite (POEQU, POEQUB) matrices.
 *
 * GEEQU first computes the largest absolute value of each row (equ_rows_kernel, one thread per row
 * and the threads of a block reading the same columns, so that the reads are coalesced), by chunks
 * of columns, then combines them in a single block per matrix (equ_rows_final_kernel), which also
 * computes ROWCND, AMAX and the row scale factors. The column maxima of diag(R) A are computed in
 * the same way (equ_cols_kernel, a group of EQU_BX threads per column, by chunks of rows, and
 * equ_cols_final_kernel). The partial results are combined in a fixed order.
 */

#ifndef EQU_BX
#define EQU_BX 64
#endif
#ifndef EQU_BY
#define EQU_BY 4
#endif
#ifndef EQU_TARGET_BLOCKS
#define EQU_TARGET_BLOCKS 2048
#endif
#ifndef EQU_MIN_CHUNK
#define EQU_MIN_CHUNK 1024 // minimum length of the chunks of rows of equ_cols_kernel
#endif
#ifndef EQU_FINAL_THDS
#define EQU_FINAL_THDS 1024
#endif

/** Absolute value used by LAPACK's equilibration routines: |re| + |im| for complex numbers. **/
template <typename T>
__device__ __forceinline__ auto equ_cabs1(const T& x)
{
    if constexpr(rocblas_is_complex<T>)
        return rocblas_abs(x.real()) + rocblas_abs(x.imag());
    else
        return rocblas_abs(x);
}

/** LOG(x) as LAPACK's xGEEQUB and xPOEQUB evaluate it. Their results depend on its last bit when x
    is (close to) a power of 2. In single precision, the logarithm is computed in double precision
    and rounded, which gives the correctly rounded value as the host libraries do; in double
    precision, log() gives the same values as the host libraries at the powers of 2. **/
template <typename S>
__device__ __forceinline__ S equ_log(const S x)
{
    if constexpr(std::is_same_v<S, float>)
        return float(log(double(x)));
    else
        return log(x);
}

/** RADIX**INT(LOG(x) / LOG(RADIX)) (POEQUB = false, as in xGEEQUB) or
    RADIX**INT(TMP * LOG(x)) with TMP = -0.5 / LOG(RADIX) (POEQUB = true, as in xPOEQUB), with
    RADIX = 2, evaluated as LAPACK does: in precision S, with the exponent truncated toward zero.
    As in LAPACK, RADIX**e with e < 0 is 1 / RADIX**(-e), which is zero if RADIX**(-e) overflows.
    x > 0. **/
template <bool POEQUB, typename S>
__device__ __forceinline__ S equ_pow2(const S x)
{
    const S logrdx = S(0x1.62e42fefa39efp-1);
    S t;
    if(POEQUB)
        t = (S(-0.5) / logrdx) * equ_log(x);
    else
        t = equ_log(x) / logrdx;
    const int e = int(t);
    if(e <= -std::numeric_limits<S>::max_exponent)
        return 0;
    return ldexp(S(1), e);
}

/** Maxima of the absolute values of each row, by chunks of columns. Grid (ceil(m / EQU_BX), ncch,
    batch_count), block (EQU_BX, EQU_BY). The result for row r and chunk h is
    part[(b * ncch + h) * m + r]. **/
template <typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(EQU_BX* EQU_BY) equ_rows_kernel(const I m,
                                                                        const I n,
                                                                        U A,
                                                                        const rocblas_stride shiftA,
                                                                        const I lda,
                                                                        const rocblas_stride strideA,
                                                                        const I cchunk,
                                                                        const I ncch,
                                                                        S* part)
{
    const I b = blockIdx.z;
    const I h = blockIdx.y;
    const I tx = threadIdx.x;
    const I ty = threadIdx.y;
    const I nrb = (m - 1) / EQU_BX + 1;

    T* a = load_ptr_batch<T>(A, b, shiftA, strideA);

    __shared__ S smax[EQU_BY][EQU_BX];

    // the indices are computed in 64 bits, as they can exceed the range of I near its end
    const int64_t cs = int64_t(h) * cchunk;
    const int64_t ce = std::min<int64_t>(cs + cchunk, n);

    for(int64_t rb = blockIdx.x; rb < nrb; rb += gridDim.x)
    {
        const int64_t r = rb * EQU_BX + tx;

        S v = 0;
        if(r < m)
        {
            int64_t c = cs + ty;
            for(; c + 3 * EQU_BY < ce; c += 4 * EQU_BY)
            {
                const S x0 = equ_cabs1(a[idx2D(r, c, lda)]);
                const S x1 = equ_cabs1(a[idx2D(r, c + EQU_BY, lda)]);
                const S x2 = equ_cabs1(a[idx2D(r, c + 2 * EQU_BY, lda)]);
                const S x3 = equ_cabs1(a[idx2D(r, c + 3 * EQU_BY, lda)]);
                v = std::max(std::max(v, x0), std::max(x1, std::max(x2, x3)));
            }
            for(; c < ce; c += EQU_BY)
                v = std::max(v, S(equ_cabs1(a[idx2D(r, c, lda)])));
        }

        smax[ty][tx] = v;
        __syncthreads();
        if(ty == 0 && r < m)
        {
            for(I k = 1; k < EQU_BY; k++)
                v = std::max(v, smax[k][tx]);
            part[(rocblas_stride(b) * ncch + h) * m + r] = v;
        }
        __syncthreads();
    }
}

/** Combines the partial maxima of the rows (ROW) or columns (!ROW) into the scale factors, and
    computes the ratio, AMAX (rows) and info, as in xGEEQU (or xGEEQUB if POW2). Grid (1, 1,
    batch_count), block EQU_FINAL_THDS. For the columns, the matrices with info != 0 are skipped,
    and info = m + j if column j is zero. **/
template <bool ROW, bool POW2, typename I, typename S>
ROCSOLVER_KERNEL void __launch_bounds__(EQU_FINAL_THDS)
    equ_final_kernel(const I m,
                     const I n,
                     const I nch,
                     const S* part,
                     S* RC,
                     const rocblas_stride strideRC,
                     S* rcnd,
                     S* amax,
                     I* info)
{
    const I b = blockIdx.z;
    const I tid = threadIdx.x;
    const int64_t len = ROW ? m : n;

    if(!ROW && info[b] != 0)
        return;

    const S smlnum = std::numeric_limits<S>::min();
    const S bignum = S(1) / smlnum;

    const S* p = part + rocblas_stride(b) * nch * len;
    S* rc = RC + b * strideRC;

    __shared__ S smin[EQU_FINAL_THDS], smx[EQU_FINAL_THDS];
    __shared__ int64_t szero[EQU_FINAL_THDS];

    // maxima, rounded to powers of 2 for xGEEQUB, and their extremes
    S vmin = bignum, vmax = 0;
    int64_t zero = len;
    for(int64_t k = tid; k < len; k += EQU_FINAL_THDS)
    {
        S v = 0;
        for(I h = 0; h < nch; h++)
            v = std::max(v, p[rocblas_stride(h) * len + k]);
        if(POW2 && v > 0)
            v = equ_pow2<false>(v);
        rc[k] = v;
        vmin = std::min(vmin, v);
        vmax = std::max(vmax, v);
        if(v == 0)
            zero = std::min(zero, k);
    }

    smin[tid] = vmin;
    smx[tid] = vmax;
    szero[tid] = zero;
    __syncthreads();
    for(I k = EQU_FINAL_THDS / 2; k > 0; k /= 2)
    {
        if(tid < k)
        {
            smin[tid] = std::min(smin[tid], smin[tid + k]);
            smx[tid] = std::max(smx[tid], smx[tid + k]);
            szero[tid] = std::min(szero[tid], szero[tid + k]);
        }
        __syncthreads();
    }
    vmin = smin[0];
    vmax = smx[0];
    zero = szero[0];

    if(ROW && tid == 0)
        amax[b] = vmax;

    if(vmin == 0)
    {
        // a zero row or column: the maxima are left in R or C, as in LAPACK
        if(tid == 0)
            info[b] = ROW ? zero + 1 : m + zero + 1;
        return;
    }

    for(int64_t k = tid; k < len; k += EQU_FINAL_THDS)
        rc[k] = S(1) / std::min(std::max(rc[k], smlnum), bignum);

    if(tid == 0)
    {
        rcnd[b] = std::max(vmin, smlnum) / std::min(vmax, bignum);
        if(ROW)
            info[b] = 0;
    }
}

/** Maxima of the absolute values of each column of diag(R) A, by chunks of rows. Grid
    (ceil(n / EQU_BY), nrch, batch_count), block (EQU_BX, EQU_BY). The matrices with info != 0 are
    skipped. The result for column c and chunk h is part[(b * nrch + h) * n + c]. **/
template <typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(EQU_BX* EQU_BY) equ_cols_kernel(const I m,
                                                                        const I n,
                                                                        U A,
                                                                        const rocblas_stride shiftA,
                                                                        const I lda,
                                                                        const rocblas_stride strideA,
                                                                        const S* R,
                                                                        const rocblas_stride strideR,
                                                                        const I* info,
                                                                        const I rchunk,
                                                                        const I nrch,
                                                                        S* part)
{
    const I b = blockIdx.z;
    const I h = blockIdx.y;
    const I tx = threadIdx.x;
    const I ty = threadIdx.y;
    const I ncb = (n - 1) / EQU_BY + 1;

    if(info[b] != 0)
        return;

    T* a = load_ptr_batch<T>(A, b, shiftA, strideA);
    const S* r = R + b * strideR;

    __shared__ S smax[EQU_BY][EQU_BX];

    // the indices are computed in 64 bits, as they can exceed the range of I near its end
    const int64_t rs = int64_t(h) * rchunk;
    const int64_t re = std::min<int64_t>(rs + rchunk, m);

    for(int64_t cb = blockIdx.x; cb < ncb; cb += gridDim.x)
    {
        const int64_t c = cb * EQU_BY + ty;

        S v = 0;
        if(c < n)
        {
            const T* ac = a + idx2D(0, c, lda);
            int64_t i = rs + tx;
            for(; i + 3 * EQU_BX < re; i += 4 * EQU_BX)
            {
                const S x0 = equ_cabs1(ac[i]) * r[i];
                const S x1 = equ_cabs1(ac[i + EQU_BX]) * r[i + EQU_BX];
                const S x2 = equ_cabs1(ac[i + 2 * EQU_BX]) * r[i + 2 * EQU_BX];
                const S x3 = equ_cabs1(ac[i + 3 * EQU_BX]) * r[i + 3 * EQU_BX];
                v = std::max(std::max(v, x0), std::max(x1, std::max(x2, x3)));
            }
            for(; i < re; i += EQU_BX)
                v = std::max(v, S(equ_cabs1(ac[i]) * r[i]));
        }

        // reduce over tx
        smax[ty][tx] = v;
        __syncthreads();
        for(I k = EQU_BX / 2; k > 0; k /= 2)
        {
            if(tx < k)
                smax[ty][tx] = std::max(smax[ty][tx], smax[ty][tx + k]);
            __syncthreads();
        }

        if(tx == 0 && c < n)
            part[(rocblas_stride(b) * nrch + h) * n + c] = smax[ty][0];
        __syncthreads();
    }
}

/** Chunks of the row and column kernels. **/
template <typename I>
void rocsolver_geequ_chunks(const I m, const I n, I* cchunk, I* ncch, I* rchunk, I* nrch)
{
    // row kernel: ceil(m / EQU_BX) blocks per chunk of columns
    const int64_t nrb = (int64_t(m) - 1) / EQU_BX + 1;
    int64_t nch = std::min((EQU_TARGET_BLOCKS - 1) / nrb + 1, (int64_t(n) - 1) / EQU_BX + 1);
    int64_t len = ((int64_t(n) - 1) / nch / EQU_BX + 1) * EQU_BX;
    *cchunk = I(len);
    *ncch = I((int64_t(n) - 1) / len + 1);

    // column kernel: ceil(n / EQU_BY) blocks per chunk of rows
    const int64_t ncb = (int64_t(n) - 1) / EQU_BY + 1;
    nch = std::min((EQU_TARGET_BLOCKS - 1) / ncb + 1, (int64_t(m) - 1) / EQU_MIN_CHUNK + 1);
    len = ((int64_t(m) - 1) / nch / EQU_BX + 1) * EQU_BX;
    *rchunk = I(len);
    *nrch = I((int64_t(m) - 1) / len + 1);
}

template <typename T, typename I, typename S>
void rocsolver_geequ_getMemorySize(const I m, const I n, const I batch_count, size_t* size_work)
{
    // if quick return no workspace needed
    if(m == 0 || n == 0 || batch_count == 0)
    {
        *size_work = 0;
        return;
    }

    // partial maxima of the rows and of the columns
    I cchunk, ncch, rchunk, nrch;
    rocsolver_geequ_chunks(m, n, &cchunk, &ncch, &rchunk, &nrch);
    *size_work = sizeof(S) * (size_t(ncch) * m + size_t(nrch) * n) * batch_count;
}

template <typename T, typename I, typename S>
rocblas_status rocsolver_geequ_argCheck(rocblas_handle handle,
                                        const I m,
                                        const I n,
                                        const I lda,
                                        T A,
                                        S* R,
                                        S* C,
                                        S* rowcnd,
                                        S* colcnd,
                                        S* amax,
                                        I* info)
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
    if((m && n && (!A || !R || !C || !rowcnd || !colcnd || !amax)) || !info)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

/** GEEQU (POW2 = false) and GEEQUB (POW2 = true). **/
template <bool POW2, typename T, typename I, typename S, typename U>
rocblas_status rocsolver_geequ_template(rocblas_handle handle,
                                        const I m,
                                        const I n,
                                        U A,
                                        const rocblas_stride shiftA,
                                        const I lda,
                                        const rocblas_stride strideA,
                                        S* R,
                                        const rocblas_stride strideR,
                                        S* C,
                                        const rocblas_stride strideC,
                                        S* rowcnd,
                                        S* colcnd,
                                        S* amax,
                                        I* info,
                                        const I batch_count,
                                        S* work)
{
    ROCSOLVER_ENTER((POW2 ? "geequb" : "geequ"), "m:", m, "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    const I blocksReset = (batch_count - 1) / BS1 + 1;
    ROCSOLVER_LAUNCH_KERNEL((reset_info<I, I, I>), dim3(blocksReset), dim3(BS1), 0, stream, info,
                            batch_count, I(0));

    // quick return: rowcnd = colcnd = 1, amax = 0
    if(m == 0 || n == 0)
    {
        if(rowcnd)
            ROCSOLVER_LAUNCH_KERNEL((reset_info<S, I, S>), dim3(blocksReset), dim3(BS1), 0, stream,
                                    rowcnd, batch_count, S(1));
        if(colcnd)
            ROCSOLVER_LAUNCH_KERNEL((reset_info<S, I, S>), dim3(blocksReset), dim3(BS1), 0, stream,
                                    colcnd, batch_count, S(1));
        if(amax)
            ROCSOLVER_LAUNCH_KERNEL((reset_info<S, I, S>), dim3(blocksReset), dim3(BS1), 0, stream,
                                    amax, batch_count, S(0));
        return rocblas_status_success;
    }

    const hipDeviceProp_t* props = rocblas_internal_get_device_prop(handle);
    const int64_t maxgrid = props->maxGridSize[0];

    I cchunk, ncch, rchunk, nrch;
    rocsolver_geequ_chunks(m, n, &cchunk, &ncch, &rchunk, &nrch);
    S* rowpart = work;
    S* colpart = work + size_t(ncch) * m * batch_count;

    dim3 threads(EQU_BX, EQU_BY, 1);
    // the kernels loop over the blocks beyond the grid; the number of threads in each dimension
    // of the grid must be less than 2^32
    const int64_t maxblocks = std::min<int64_t>(maxgrid, ((int64_t(1) << 32) - 1) / EQU_BX);
    const I nrb = I(std::min<int64_t>((int64_t(m) - 1) / EQU_BX + 1, maxblocks));
    const I ncb = I(std::min<int64_t>((int64_t(n) - 1) / EQU_BY + 1, maxblocks));

    // row scale factors, rowcnd, amax, info (zero rows)
    ROCSOLVER_LAUNCH_KERNEL((equ_rows_kernel<T>), dim3(nrb, ncch, batch_count), threads, 0, stream,
                            m, n, A, shiftA, lda, strideA, cchunk, ncch, rowpart);
    ROCSOLVER_LAUNCH_KERNEL((equ_final_kernel<true, POW2, I, S>), dim3(1, 1, batch_count),
                            dim3(EQU_FINAL_THDS), 0, stream, m, n, ncch, rowpart, R, strideR,
                            rowcnd, amax, info);

    // column scale factors, colcnd, info (zero columns)
    ROCSOLVER_LAUNCH_KERNEL((equ_cols_kernel<T>), dim3(ncb, nrch, batch_count), threads, 0, stream,
                            m, n, A, shiftA, lda, strideA, R, strideR, info, rchunk, nrch, colpart);
    ROCSOLVER_LAUNCH_KERNEL((equ_final_kernel<false, POW2, I, S>), dim3(1, 1, batch_count),
                            dim3(EQU_FINAL_THDS), 0, stream, m, n, nrch, colpart, C, strideC,
                            colcnd, amax, info);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE

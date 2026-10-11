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
 * Norms of general (LANGE), trapezoidal (LANTR), symmetric (LANSY) and Hermitian (LANHE)
 * matrices.
 *
 * Only the stored part of the matrix is read. The column sums, maxima and scaled sums of squares are
 * computed by lan_cols_kernel (each group of LAN_BX threads goes down one column, so the reads
 * are coalesced), and the row sums by lan_rows_kernel (each thread takes one row and the threads
 * of a block read the same columns, so these reads are coalesced too). The columns (rows) are
 * split into chunks so that the grid has about LAN_TARGET_BLOCKS blocks; each chunk writes a
 * partial result, and lan_final_kernel adds the partial results in a fixed order (the result does
 * not depend on the scheduling of the blocks).
 *
 * For a symmetric or Hermitian matrix stored in the upper (lower) triangle, the sum of column k of
 * the full matrix is the sum of column k of the stored triangle, including the diagonal, plus the
 * sum of row k of the stored triangle, excluding the diagonal.
 *
 * The Frobenius norm uses the scaled sums of squares of LAPACK's xLASSQ (Blue's algorithm): each
 * element goes into one of three accumulators (big, medium, small) depending on its magnitude, so
 * that the sum of squares does not overflow or underflow unless the norm itself does. As in
 * xLASSQ, the real and imaginary parts of complex elements are taken separately.
 */

#define LAN_KIND_TR 0
#define LAN_KIND_SY 1
#define LAN_KIND_HE 2
#define LAN_KIND_GE 3

// symmetric or Hermitian: only one triangle is stored
#define LAN_SYMMETRIC(KIND) ((KIND) == LAN_KIND_SY || (KIND) == LAN_KIND_HE)

#define LAN_NORM_SUM 0
#define LAN_NORM_MAX 1
#define LAN_NORM_FRO 2

#ifndef LAN_BX
#define LAN_BX 64
#endif
#ifndef LAN_BY
#define LAN_BY 4
#endif
#ifndef LAN_TARGET_BLOCKS
#define LAN_TARGET_BLOCKS 2048
#endif
#ifndef LAN_MIN_CHUNK
#define LAN_MIN_CHUNK 1024 // minimum length of the chunks of rows of lan_cols_kernel
#endif
#ifndef LAN_FINAL_THDS
#define LAN_FINAL_THDS 1024
#endif

/** Thresholds and scaling constants of Blue's algorithm (LAPACK's la_constants). **/
template <typename S>
struct lan_blue;

template <>
struct lan_blue<float>
{
    static constexpr float tsml = 0x1p-63f;
    static constexpr float tbig = 0x1p52f;
    static constexpr float ssml = 0x1p75f;
    static constexpr float sbig = 0x1p-76f;
};

template <>
struct lan_blue<double>
{
    static constexpr double tsml = 0x1p-511;
    static constexpr double tbig = 0x1p486;
    static constexpr double ssml = 0x1p537;
    static constexpr double sbig = 0x1p-538;
};

/** Adds w * x^2 to the accumulator of Blue's algorithm that corresponds to |x| (NaN goes to the
    medium accumulator, as in xLASSQ). Written without branches, so that the accumulators stay in
    registers. **/
template <typename S>
__device__ __forceinline__ void lan_ssq_add(const S ax, const S w, S& abig, S& amed, S& asml)
{
    const bool big = ax > lan_blue<S>::tbig;
    const bool sml = ax < lan_blue<S>::tsml;
    const S xb = big ? ax * lan_blue<S>::sbig : S(0);
    const S xs = sml ? ax * lan_blue<S>::ssml : S(0);
    const S xm = (big || sml) ? S(0) : ax;
    abig += w * xb * xb;
    amed += w * xm * xm;
    asml += w * xs * xs;
}

/** Square root of the sum of squares given by the three accumulators (as in xLASSQ). **/
template <typename S>
__device__ __forceinline__ S lan_ssq_combine(S abig, S amed, S asml)
{
    S scl, sumsq;
    if(abig > 0)
    {
        if(amed > 0 || rocblas_isnan(amed))
            abig += (amed * lan_blue<S>::sbig) * lan_blue<S>::sbig;
        scl = S(1) / lan_blue<S>::sbig;
        sumsq = abig;
    }
    else if(asml > 0)
    {
        if(amed > 0 || rocblas_isnan(amed))
        {
            amed = sqrt(amed);
            asml = sqrt(asml) / lan_blue<S>::ssml;
            S ymin = asml > amed ? amed : asml;
            S ymax = asml > amed ? asml : amed;
            scl = 1;
            sumsq = ymax * ymax * (1 + (ymin / ymax) * (ymin / ymax));
        }
        else
        {
            scl = S(1) / lan_blue<S>::ssml;
            sumsq = asml;
        }
    }
    else
    {
        scl = 1;
        sumsq = amed;
    }
    return scl * sqrt(sumsq);
}

/** Absolute value of a complex element that is NaN if either part is NaN. **/
template <typename T>
__device__ __forceinline__ auto lan_abs(const T& x)
{
    using S = decltype(std::real(x));
    if constexpr(rocblas_is_complex<T>)
        return rocblas_isnan(x) ? std::numeric_limits<S>::quiet_NaN() : S(rocblas_abs(x));
    else
        return rocblas_abs(x);
}

/** Range [lo, hi) of rows of the stored part of column c that lan_cols_kernel visits. For a
    symmetric/Hermitian matrix, the diagonal is included. **/
template <int KIND>
__device__ __forceinline__ void
    lan_col_range(const bool upper, const int64_t m, const int64_t c, int64_t& lo, int64_t& hi)
{
    if(KIND == LAN_KIND_GE)
    {
        lo = 0;
        hi = m;
    }
    else if(upper)
    {
        lo = 0;
        hi = std::min(c + 1, m);
    }
    else
    {
        lo = c;
        hi = m;
    }
}

/** Range [lo, hi) of columns of the stored part of row r that lan_rows_kernel visits. For a
    symmetric/Hermitian matrix, the diagonal is excluded (it is counted by the column sums). **/
template <int KIND>
__device__ __forceinline__ void
    lan_row_range(const bool upper, const int64_t n, const int64_t r, int64_t& lo, int64_t& hi)
{
    constexpr int64_t d = LAN_SYMMETRIC(KIND) ? 1 : 0;
    if(KIND == LAN_KIND_GE)
    {
        lo = 0;
        hi = n;
    }
    else if(upper)
    {
        lo = r + d;
        hi = n;
    }
    else
    {
        lo = 0;
        hi = std::min(r + 1 - d, n);
    }
}

/** Adds element x = A(r, c) to the column accumulators of lan_cols_kernel. **/
template <int KIND, int NORM, typename T, typename I, typename S>
__device__ __forceinline__ void
    lan_col_add(const bool unit, const I r, const I c, const T x, S& acc, S& abig, S& asml)
{
    const bool isdiag = (r == c);

    if constexpr(NORM == LAN_NORM_FRO)
    {
        // off-diagonal elements of a symmetric/Hermitian matrix are counted twice
        const S w = (LAN_SYMMETRIC(KIND) && !isdiag) ? S(2) : S(1);
        if(KIND == LAN_KIND_TR && unit && isdiag)
            acc += 1;
        else if constexpr(rocblas_is_complex<T>)
        {
            lan_ssq_add<S>(rocblas_abs(x.real()), w, abig, acc, asml);
            if(KIND != LAN_KIND_HE || !isdiag)
                lan_ssq_add<S>(rocblas_abs(x.imag()), w, abig, acc, asml);
        }
        else
            lan_ssq_add<S>(rocblas_abs(x), w, abig, acc, asml);
    }
    else
    {
        S v;
        if(KIND == LAN_KIND_TR && unit && isdiag)
            v = 1;
        else if(KIND == LAN_KIND_HE && isdiag)
            v = rocblas_abs(std::real(x));
        else
            v = lan_abs(x);

        if constexpr(NORM == LAN_NORM_SUM)
            acc += v;
        else
            acc = rocblas_max_nan(acc, v);
    }
}

/** Sum, maximum or scaled sums of squares of the stored elements of each column, by chunks of
    rows. Grid (ceil(n / LAN_BY), nrch, batch_count), block (LAN_BX, LAN_BY): thread (tx, ty) visits
    rows tx, tx + LAN_BX, ... of column blockIdx.x * LAN_BY + ty within the chunk blockIdx.y.
    The result for column c and chunk h is part[(b * nrch + h) * n + c] (and, for the Frobenius norm,
    the three accumulators are in part, part + psize and part + 2 * psize). **/
template <int KIND, int NORM, typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(LAN_BX* LAN_BY) lan_cols_kernel(const bool upper,
                                                                        const bool unit,
                                                                        const I m,
                                                                        const I n,
                                                                        U A,
                                                                        const rocblas_stride shiftA,
                                                                        const I lda,
                                                                        const rocblas_stride strideA,
                                                                        const I rchunk,
                                                                        const I nrch,
                                                                        S* part,
                                                                        const rocblas_stride psize)
{
    const I b = blockIdx.z;
    const I h = blockIdx.y;
    const I tx = threadIdx.x;
    const I ty = threadIdx.y;
    const I ncb = (n - 1) / LAN_BY + 1;

    T* a = load_ptr_batch<T>(A, b, shiftA, strideA);

    // the medium accumulator (or the sum or the maximum), then the big and small accumulators
    __shared__ S sacc[3][LAN_BY][LAN_BX];

    // the indices are computed in 64 bits, as they can exceed the range of I near its end
    const int64_t rs = int64_t(h) * rchunk;
    const int64_t re = std::min<int64_t>(rs + rchunk, m);

    for(int64_t cb = blockIdx.x; cb < ncb; cb += gridDim.x)
    {
        const int64_t c = cb * LAN_BY + ty;
        S acc = 0, abig = 0, asml = 0;

        if(c < n)
        {
            int64_t lo, hi;
            lan_col_range<KIND>(upper, m, c, lo, hi);
            lo = std::max(lo, rs);
            hi = std::min(hi, re);

            // four independent loads per iteration, then the remaining rows
            const T* ac = a + idx2D(0, c, lda);
            int64_t r = lo + tx;
            for(; r + 3 * LAN_BX < hi; r += 4 * LAN_BX)
            {
                const T x0 = ac[r];
                const T x1 = ac[r + LAN_BX];
                const T x2 = ac[r + 2 * LAN_BX];
                const T x3 = ac[r + 3 * LAN_BX];
                lan_col_add<KIND, NORM, T, int64_t, S>(unit, r, c, x0, acc, abig, asml);
                lan_col_add<KIND, NORM, T, int64_t, S>(unit, r + LAN_BX, c, x1, acc, abig, asml);
                lan_col_add<KIND, NORM, T, int64_t, S>(unit, r + 2 * LAN_BX, c, x2, acc, abig, asml);
                lan_col_add<KIND, NORM, T, int64_t, S>(unit, r + 3 * LAN_BX, c, x3, acc, abig, asml);
            }
            for(; r < hi; r += LAN_BX)
                lan_col_add<KIND, NORM, T, int64_t, S>(unit, r, c, ac[r], acc, abig, asml);
        }

        // reduce over tx (in a fixed order)
        sacc[0][ty][tx] = acc;
        if constexpr(NORM == LAN_NORM_FRO)
        {
            sacc[1][ty][tx] = abig;
            sacc[2][ty][tx] = asml;
        }
        __syncthreads();
        for(I k = LAN_BX / 2; k > 0; k /= 2)
        {
            if(tx < k)
            {
                if constexpr(NORM == LAN_NORM_MAX)
                    sacc[0][ty][tx] = rocblas_max_nan(sacc[0][ty][tx], sacc[0][ty][tx + k]);
                else
                    sacc[0][ty][tx] += sacc[0][ty][tx + k];
                if constexpr(NORM == LAN_NORM_FRO)
                {
                    sacc[1][ty][tx] += sacc[1][ty][tx + k];
                    sacc[2][ty][tx] += sacc[2][ty][tx + k];
                }
            }
            __syncthreads();
        }

        if(tx == 0 && c < n)
        {
            const rocblas_stride p = (rocblas_stride(b) * nrch + h) * n + c;
            part[p] = sacc[0][ty][0];
            if constexpr(NORM == LAN_NORM_FRO)
            {
                part[p + psize] = sacc[1][ty][0];
                part[p + 2 * psize] = sacc[2][ty][0];
            }
        }
        __syncthreads();
    }
}

/** Sums of the absolute values of the stored elements of each row (excluding the diagonal for a
    symmetric/Hermitian matrix), by chunks of columns. Grid (ceil(m / LAN_BX), ncch, batch_count),
    block (LAN_BX, LAN_BY): thread (tx, ty) takes row blockIdx.x * LAN_BX + tx and the columns
    ty, ty + LAN_BY, ... of the chunk blockIdx.y. The result for row r and chunk h is
    part[(b * ncch + h) * m + r]. **/
template <int KIND, typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(LAN_BX* LAN_BY) lan_rows_kernel(const bool upper,
                                                                        const bool unit,
                                                                        const I m,
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
    const I nrb = (m - 1) / LAN_BX + 1;

    T* a = load_ptr_batch<T>(A, b, shiftA, strideA);

    __shared__ S ssum[LAN_BY][LAN_BX];

    // the indices are computed in 64 bits, as they can exceed the range of I near its end
    const int64_t cs = int64_t(h) * cchunk;
    const int64_t ce = std::min<int64_t>(cs + cchunk, n);

    for(int64_t rb = blockIdx.x; rb < nrb; rb += gridDim.x)
    {
        const int64_t r0 = rb * LAN_BX;
        const int64_t r = r0 + tx;
        const int64_t rlast = std::min<int64_t>(r0 + LAN_BX, m) - 1;

        // the columns visited by the block (the same for all its threads, so that the reads of
        // each column are coalesced); each thread only adds those of its own row. The start of
        // the range of a row does not decrease with the row, and neither does its end.
        int64_t lo, hi, rlo, rhi;
        lan_row_range<KIND>(upper, n, r0, lo, rhi);
        lan_row_range<KIND>(upper, n, rlast, rlo, hi);
        lo = std::max(lo, cs);
        hi = std::min(hi, ce);
        if(r <= rlast)
            lan_row_range<KIND>(upper, n, r, rlo, rhi);
        else
            rlo = rhi = 0;

        // adds the element of column c (if it is in the row)
        S sum = 0;
        auto add = [&](const int64_t c) __attribute__((always_inline))
        {
            if(c >= rlo && c < rhi)
            {
                if(KIND == LAN_KIND_TR && unit && c == r)
                    sum += 1;
                else
                    sum += lan_abs(a[idx2D(r, c, lda)]);
            }
        };

        // four columns per iteration, then the remaining columns
        int64_t c = lo + ty;
        for(; c + 3 * LAN_BY < hi; c += 4 * LAN_BY)
        {
            add(c);
            add(c + LAN_BY);
            add(c + 2 * LAN_BY);
            add(c + 3 * LAN_BY);
        }
        for(; c < hi; c += LAN_BY)
            add(c);

        ssum[ty][tx] = sum;
        __syncthreads();
        if(ty == 0 && r <= rlast)
        {
            for(I k = 1; k < LAN_BY; k++)
                sum += ssum[k][tx];
            part[(rocblas_stride(b) * ncch + h) * m + r] = sum;
        }
        __syncthreads();
    }
}

/** Combines the partial results into the norm of each matrix. Grid (1, 1, batch_count), block
    LAN_FINAL_THDS. For the 1-norm (infinity-norm) of a trapezoidal matrix, colpart (rowpart) holds
    the column (row) sums; for a symmetric/Hermitian matrix, both are used. **/
template <int KIND, int NORM, typename I, typename S>
ROCSOLVER_KERNEL void __launch_bounds__(LAN_FINAL_THDS)
    lan_final_kernel(const rocsolver_norm_type norm_type,
                     const I m,
                     const I n,
                     const I nrch,
                     const S* colpart,
                     const rocblas_stride psize,
                     const I ncch,
                     const S* rowpart,
                     S* norms)
{
    const I b = blockIdx.z;
    const I tid = threadIdx.x;

    __shared__ S sacc[3][LAN_FINAL_THDS];

    S acc = 0, abig = 0, asml = 0;

    if constexpr(NORM == LAN_NORM_SUM)
    {
        const bool rows = (!LAN_SYMMETRIC(KIND) && norm_type == rocsolver_norm_type_infinity);
        const int64_t len = rows ? m : n;
        const S* cp = colpart + rocblas_stride(b) * nrch * n;
        const S* rp = rowpart + rocblas_stride(b) * ncch * m;
        for(int64_t k = tid; k < len; k += LAN_FINAL_THDS)
        {
            S sum = 0;
            if(!rows)
                for(I h = 0; h < nrch; h++)
                    sum += cp[rocblas_stride(h) * n + k];
            if(rows || LAN_SYMMETRIC(KIND))
                for(I h = 0; h < ncch; h++)
                    sum += rp[rocblas_stride(h) * m + k];
            acc = rocblas_max_nan(acc, sum);
        }
    }
    else
    {
        const rocblas_stride len = rocblas_stride(nrch) * n;
        const S* cp = colpart + b * len;
        for(rocblas_stride k = tid; k < len; k += LAN_FINAL_THDS)
        {
            if constexpr(NORM == LAN_NORM_MAX)
                acc = rocblas_max_nan(acc, cp[k]);
            else
            {
                acc += cp[k];
                abig += cp[k + psize];
                asml += cp[k + 2 * psize];
            }
        }
    }

    sacc[0][tid] = acc;
    if constexpr(NORM == LAN_NORM_FRO)
    {
        sacc[1][tid] = abig;
        sacc[2][tid] = asml;
    }
    __syncthreads();
    for(I k = LAN_FINAL_THDS / 2; k > 0; k /= 2)
    {
        if(tid < k)
        {
            if constexpr(NORM == LAN_NORM_FRO)
            {
                sacc[0][tid] += sacc[0][tid + k];
                sacc[1][tid] += sacc[1][tid + k];
                sacc[2][tid] += sacc[2][tid + k];
            }
            else
                sacc[0][tid] = rocblas_max_nan(sacc[0][tid], sacc[0][tid + k]);
        }
        __syncthreads();
    }

    if(tid == 0)
    {
        if constexpr(NORM == LAN_NORM_FRO)
            norms[b] = lan_ssq_combine<S>(sacc[1][0], sacc[0][0], sacc[2][0]);
        else
            norms[b] = sacc[0][0];
    }
}

/** Chunk sizes and number of chunks of the column (row) kernels, and the size of the partial
    results. **/
template <int KIND, typename I>
void rocsolver_lan_chunks(const rocsolver_norm_type norm_type,
                          const I m,
                          const I n,
                          I* rchunk,
                          I* nrch,
                          I* cchunk,
                          I* ncch,
                          bool* need_cols,
                          bool* need_rows)
{
    const bool sum
        = (norm_type == rocsolver_norm_type_one || norm_type == rocsolver_norm_type_infinity);
    *need_rows = sum && (LAN_SYMMETRIC(KIND) || norm_type == rocsolver_norm_type_infinity);
    *need_cols = !sum || LAN_SYMMETRIC(KIND) || norm_type == rocsolver_norm_type_one;

    // column kernel: ceil(n / LAN_BY) blocks per chunk of rows
    const int64_t ncb = (int64_t(n) - 1) / LAN_BY + 1;
    int64_t nch = std::min((LAN_TARGET_BLOCKS - 1) / ncb + 1, (int64_t(m) - 1) / LAN_MIN_CHUNK + 1);
    int64_t len = ((int64_t(m) - 1) / nch / LAN_BX + 1) * LAN_BX;
    *rchunk = I(len);
    *nrch = I((int64_t(m) - 1) / len + 1);

    // row kernel: ceil(m / LAN_BX) blocks per chunk of columns
    const int64_t nrb = (int64_t(m) - 1) / LAN_BX + 1;
    nch = std::min((LAN_TARGET_BLOCKS - 1) / nrb + 1, (int64_t(n) - 1) / LAN_BX + 1);
    len = ((int64_t(n) - 1) / nch / LAN_BX + 1) * LAN_BX;
    *cchunk = I(len);
    *ncch = I((int64_t(n) - 1) / len + 1);
}

template <int KIND, typename T, typename I, typename S>
void rocsolver_lan_getMemorySize(const rocsolver_norm_type norm_type,
                                 const I m,
                                 const I n,
                                 const I batch_count,
                                 size_t* size_work)
{
    // if quick return no workspace needed
    if(m == 0 || n == 0 || batch_count == 0)
    {
        *size_work = 0;
        return;
    }

    I rchunk, nrch, cchunk, ncch;
    bool need_cols, need_rows;
    rocsolver_lan_chunks<KIND>(norm_type, m, n, &rchunk, &nrch, &cchunk, &ncch, &need_cols,
                               &need_rows);

    const size_t nacc = (norm_type == rocsolver_norm_type_frobenius) ? 3 : 1;
    size_t size = 0;
    if(need_cols)
        size += nacc * size_t(nrch) * n;
    if(need_rows)
        size += size_t(ncch) * m;
    *size_work = sizeof(S) * size * batch_count;
}

/** Computes the norms of a batch of trapezoidal, symmetric or Hermitian matrices (KIND). For a
    symmetric/Hermitian matrix, m = n and diag is not used. **/
template <int KIND, typename T, typename I, typename S, typename U>
rocblas_status rocsolver_lan_template(rocblas_handle handle,
                                      const rocsolver_norm_type norm_type,
                                      const rocblas_fill uplo,
                                      const rocblas_diagonal diag,
                                      const I m,
                                      const I n,
                                      U A,
                                      const rocblas_stride shiftA,
                                      const I lda,
                                      const rocblas_stride strideA,
                                      const I batch_count,
                                      S* norms,
                                      S* work)
{
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // quick return: the norm of an empty matrix is zero
    if(m == 0 || n == 0)
    {
        if(norms)
        {
            const I blocks = (batch_count - 1) / BS1 + 1;
            ROCSOLVER_LAUNCH_KERNEL((reset_info<S, I, S>), dim3(blocks), dim3(BS1), 0, stream,
                                    norms, batch_count, S(0));
        }
        return rocblas_status_success;
    }

    const hipDeviceProp_t* props = rocblas_internal_get_device_prop(handle);
    const int64_t maxgrid = props->maxGridSize[0];

    const bool upper = (uplo == rocblas_fill_upper);
    const bool unit = (KIND == LAN_KIND_TR && diag == rocblas_diagonal_unit);

    I rchunk, nrch, cchunk, ncch;
    bool need_cols, need_rows;
    rocsolver_lan_chunks<KIND>(norm_type, m, n, &rchunk, &nrch, &cchunk, &ncch, &need_cols,
                               &need_rows);

    // partial results: the column results (three arrays of psize elements for the Frobenius
    // norm), then the row sums
    const rocblas_stride psize = rocblas_stride(nrch) * n * batch_count;
    const size_t nacc = (norm_type == rocsolver_norm_type_frobenius) ? 3 : 1;
    S* colpart = work;
    S* rowpart = work + (need_cols ? nacc * psize : 0);

    dim3 threads(LAN_BX, LAN_BY, 1);
    // the kernels loop over the blocks beyond the grid; the number of threads in each dimension
    // of the grid must be less than 2^32
    const int64_t maxblocks = std::min<int64_t>(maxgrid, ((int64_t(1) << 32) - 1) / LAN_BX);
    const I ncb = I(std::min<int64_t>((int64_t(n) - 1) / LAN_BY + 1, maxblocks));
    const I nrb = I(std::min<int64_t>((int64_t(m) - 1) / LAN_BX + 1, maxblocks));
    dim3 gridc(ncb, nrch, batch_count);
    dim3 gridr(nrb, ncch, batch_count);
    dim3 gridf(1, 1, batch_count);

    switch(norm_type)
    {
    case rocsolver_norm_type_one:
    case rocsolver_norm_type_infinity:
        if(need_cols)
            ROCSOLVER_LAUNCH_KERNEL((lan_cols_kernel<KIND, LAN_NORM_SUM, T>), gridc, threads, 0,
                                    stream, upper, unit, m, n, A, shiftA, lda, strideA, rchunk,
                                    nrch, colpart, psize);
        if(need_rows)
            ROCSOLVER_LAUNCH_KERNEL((lan_rows_kernel<KIND, T>), gridr, threads, 0, stream, upper,
                                    unit, m, n, A, shiftA, lda, strideA, cchunk, ncch, rowpart);
        ROCSOLVER_LAUNCH_KERNEL((lan_final_kernel<KIND, LAN_NORM_SUM, I, S>), gridf,
                                dim3(LAN_FINAL_THDS), 0, stream, norm_type, m, n, nrch, colpart,
                                psize, ncch, rowpart, norms);
        break;
    case rocsolver_norm_type_max:
        ROCSOLVER_LAUNCH_KERNEL((lan_cols_kernel<KIND, LAN_NORM_MAX, T>), gridc, threads, 0, stream,
                                upper, unit, m, n, A, shiftA, lda, strideA, rchunk, nrch, colpart,
                                psize);
        ROCSOLVER_LAUNCH_KERNEL((lan_final_kernel<KIND, LAN_NORM_MAX, I, S>), gridf,
                                dim3(LAN_FINAL_THDS), 0, stream, norm_type, m, n, nrch, colpart,
                                psize, ncch, rowpart, norms);
        break;
    case rocsolver_norm_type_frobenius:
        ROCSOLVER_LAUNCH_KERNEL((lan_cols_kernel<KIND, LAN_NORM_FRO, T>), gridc, threads, 0, stream,
                                upper, unit, m, n, A, shiftA, lda, strideA, rchunk, nrch, colpart,
                                psize);
        ROCSOLVER_LAUNCH_KERNEL((lan_final_kernel<KIND, LAN_NORM_FRO, I, S>), gridf,
                                dim3(LAN_FINAL_THDS), 0, stream, norm_type, m, n, nrch, colpart,
                                psize, ncch, rowpart, norms);
        break;
    default: return rocblas_status_invalid_value;
    }

    return rocblas_status_success;
}

/*************************************************************
    LANTR
*************************************************************/

template <typename T, typename I, typename S>
void rocsolver_lantr_getMemorySize(const rocsolver_norm_type norm_type,
                                   const I m,
                                   const I n,
                                   const I batch_count,
                                   size_t* size_work)
{
    rocsolver_lan_getMemorySize<LAN_KIND_TR, T, I, S>(norm_type, m, n, batch_count, size_work);
}

template <typename T, typename I, typename S>
rocblas_status rocsolver_lantr_argCheck(rocblas_handle handle,
                                        const rocsolver_norm_type norm_type,
                                        const rocblas_fill uplo,
                                        const rocblas_diagonal diag,
                                        const I m,
                                        const I n,
                                        const I lda,
                                        T A,
                                        S* norm)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(norm_type != rocsolver_norm_type_one && norm_type != rocsolver_norm_type_frobenius
       && norm_type != rocsolver_norm_type_infinity && norm_type != rocsolver_norm_type_max)
        return rocblas_status_invalid_value;
    if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
        return rocblas_status_invalid_value;
    if(diag != rocblas_diagonal_non_unit && diag != rocblas_diagonal_unit)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(m < 0 || n < 0 || lda < m)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((m && n && !A) || (m && n && !norm))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename I, typename S, typename U>
rocblas_status rocsolver_lantr_template(rocblas_handle handle,
                                        const rocsolver_norm_type norm_type,
                                        const rocblas_fill uplo,
                                        const rocblas_diagonal diag,
                                        const I m,
                                        const I n,
                                        U A,
                                        const rocblas_stride shiftA,
                                        const I lda,
                                        const rocblas_stride strideA,
                                        const I batch_count,
                                        S* norms,
                                        S* work)
{
    ROCSOLVER_ENTER("lantr", "norm_type:", norm_type, "uplo:", uplo, "diag:", diag, "m:", m,
                    "n:", n, "shiftA:", shiftA, "lda:", lda, "bc:", batch_count);

    return rocsolver_lan_template<LAN_KIND_TR, T>(handle, norm_type, uplo, diag, m, n, A, shiftA,
                                                  lda, strideA, batch_count, norms, work);
}

ROCSOLVER_END_NAMESPACE

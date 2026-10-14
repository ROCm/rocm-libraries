/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routines (version 3.12.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2023
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

/*************************************************************
 * The column sums, maxima and scaled sums of squares are computed by lange_cols_kernel (each group
 * of bx threads goes down one column, so the reads are coalesced), and the row sums of the
 * infinity-norm by lange_rows_kernel (each thread takes one row and the threads of a block read
 * the same columns, so these reads are coalesced too). bx is smaller than LANGE_BX when the
 * matrix has few rows (down to one thread per column), and is LANGE_THDS when it has few
 * columns, so that the threads are not idle. The columns (rows) are split into chunks so that the
 * grid has about LANGE_TARGET_BLOCKS blocks; each chunk writes partial results, which
 * lange_final_kernel combines in a fixed order (the result does not depend on the scheduling of
 * the blocks). Unless the column sums of the 1-norm (the row sums of the infinity-norm) are split
 * into chunks, a block combines the results of its columns (rows), so that there is one partial
 * result per block.
 *
 * The Frobenius norm uses the scaled sums of squares of LAPACK's xLASSQ (Blue's algorithm): each
 * element goes into one of three accumulators (big, medium, small) depending on its magnitude, so
 * that the sum of squares does not overflow or underflow unless the norm itself does. As in
 * xLASSQ, the real and imaginary parts of complex elements are taken separately. When the partial
 * results are added, a medium accumulator that has grown past tmed is moved into the big one
 * (scaled), as it could otherwise overflow while the norm is representable.
 *************************************************************/

#define LANGE_NORM_SUM 0
#define LANGE_NORM_MAX 1
#define LANGE_NORM_FRO 2

#ifndef LANGE_THDS
#define LANGE_THDS 256 // threads per block of the column and row kernels
#endif
#ifndef LANGE_BX
#define LANGE_BX 64 // threads per column of the column kernel (when m >= 2 * LANGE_BX)
#endif
// maximum size of the first dimension of the grids of the column and row kernels
#ifndef LANGE_MAX_BLOCKS
#define LANGE_MAX_BLOCKS 8192
#endif
#ifndef LANGE_TARGET_BLOCKS
#define LANGE_TARGET_BLOCKS 2048
#endif
#ifndef LANGE_MIN_CHUNK
#define LANGE_MIN_CHUNK 1024 // minimum length of the chunks of rows of lange_cols_kernel
#endif
#ifndef LANGE_FINAL_THDS
#define LANGE_FINAL_THDS 1024
#endif

/** Thresholds and scaling constants of Blue's algorithm (LAPACK's la_constants). **/
template <typename S>
struct lange_blue;

template <>
struct lange_blue<float>
{
    static constexpr float tsml = 0x1p-63f;
    static constexpr float tbig = 0x1p52f;
    static constexpr float ssml = 0x1p75f;
    static constexpr float sbig = 0x1p-76f;
    static constexpr float tmed = 0x1p126f;
};

template <>
struct lange_blue<double>
{
    static constexpr double tsml = 0x1p-511;
    static constexpr double tbig = 0x1p486;
    static constexpr double ssml = 0x1p537;
    static constexpr double sbig = 0x1p-538;
    static constexpr double tmed = 0x1p1022;
};

/** Adds x^2 to the accumulator of Blue's algorithm that corresponds to ax = |x| (NaN goes to the
    medium accumulator, as in xLASSQ). Written without branches, so that the accumulators stay in
    registers. **/
template <typename S>
__device__ __forceinline__ void lange_ssq_add(const S ax, S& abig, S& amed, S& asml)
{
    const bool big = ax > lange_blue<S>::tbig;
    const bool sml = ax < lange_blue<S>::tsml;
    const S xb = big ? ax * lange_blue<S>::sbig : S(0);
    const S xs = sml ? ax * lange_blue<S>::ssml : S(0);
    const S xm = (big || sml) ? S(0) : ax;
    abig += xb * xb;
    amed += xm * xm;
    asml += xs * xs;
}

/** Moves the medium accumulator into the big one if it exceeds tmed. **/
template <typename S>
__device__ __forceinline__ void lange_ssq_renorm(S& abig, S& amed)
{
    if(amed > lange_blue<S>::tmed)
    {
        abig += (amed * lange_blue<S>::sbig) * lange_blue<S>::sbig;
        amed = 0;
    }
}

/** Adds the partial accumulators (bbig, bmed, bsml) to (abig, amed, asml). The medium
    accumulators are at most tmed, so their sum does not overflow; if it exceeds tmed, it is moved
    into the big accumulator. **/
template <typename S>
__device__ __forceinline__ void
    lange_ssq_merge(S& abig, S& amed, S& asml, const S bbig, const S bmed, const S bsml)
{
    abig += bbig;
    amed += bmed;
    asml += bsml;
    lange_ssq_renorm<S>(abig, amed);
}

/** Square root of the sum of squares given by the three accumulators (as in xLASSQ). **/
template <typename S>
__device__ __forceinline__ S lange_ssq_combine(S abig, S amed, S asml)
{
    S scl, sumsq;
    if(abig > 0)
    {
        if(amed > 0 || rocblas_isnan(amed))
            abig += (amed * lange_blue<S>::sbig) * lange_blue<S>::sbig;
        scl = S(1) / lange_blue<S>::sbig;
        sumsq = abig;
    }
    else if(asml > 0)
    {
        if(amed > 0 || rocblas_isnan(amed))
        {
            amed = sqrt(amed);
            asml = sqrt(asml) / lange_blue<S>::ssml;
            S ymin = asml > amed ? amed : asml;
            S ymax = asml > amed ? asml : amed;
            scl = 1;
            sumsq = ymax * ymax * (1 + (ymin / ymax) * (ymin / ymax));
        }
        else
        {
            scl = S(1) / lange_blue<S>::ssml;
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

/** Absolute value of an element; for a complex element, NaN if either part is NaN. **/
template <typename T>
__device__ __forceinline__ auto lange_abs(const T& x)
{
    using S = decltype(std::real(x));
    if constexpr(rocblas_is_complex<T>)
        return rocblas_isnan(x) ? std::numeric_limits<S>::quiet_NaN() : S(rocblas_abs(x));
    else
        return rocblas_abs(x);
}

/** Adds element x to the column accumulators of lange_cols_kernel. **/
template <int NORM, typename T, typename S>
__device__ __forceinline__ void lange_col_add(const T x, S& acc, S& abig, S& asml)
{
    if constexpr(NORM == LANGE_NORM_FRO)
    {
        if constexpr(rocblas_is_complex<T>)
        {
            lange_ssq_add<S>(rocblas_abs(x.real()), abig, acc, asml);
            lange_ssq_add<S>(rocblas_abs(x.imag()), abig, acc, asml);
        }
        else
            lange_ssq_add<S>(rocblas_abs(x), abig, acc, asml);
    }
    else if constexpr(NORM == LANGE_NORM_SUM)
        acc += lange_abs(x);
    else
        acc = rocblas_max_nan(acc, lange_abs(x));
}

/** Number of threads per column of lange_cols_kernel, and of rows per block of lange_rows_kernel,
    for m-by-n matrices (the LANGE_THDS / bx threads in the other dimension of the block take
    different columns), so that the threads are not idle when there are few rows or few
    columns. **/
inline int lange_bx(const int64_t m, const int64_t n)
{
    if(n <= LANGE_THDS / LANGE_BX && m >= 2 * LANGE_THDS)
        return LANGE_THDS;
    return m >= 2 * LANGE_BX ? LANGE_BX : (m >= 64 ? 16 : (m >= 16 ? 4 : 1));
}

/** Combines the result of a column (or of a group of columns) with that of other columns: the
    maximum of the column sums (1-norm) or of the maxima, or the sum of the sums of squares. **/
template <int NORM, typename S>
__device__ __forceinline__ void
    lange_cols_combine(S& acc, S& abig, S& asml, const S bacc, const S bbig, const S bsml)
{
    if constexpr(NORM == LANGE_NORM_FRO)
        lange_ssq_merge<S>(abig, acc, asml, bbig, bacc, bsml);
    else
        acc = rocblas_max_nan(acc, bacc);
}

/** Adds the elements of rows r0, r0 + BX, ... (below re) of column ac to the accumulators. **/
template <int NORM, int BX, typename T, typename S>
__device__ __forceinline__ void
    lange_col_rows(const T* ac, const int64_t r0, const int64_t re, S& acc, S& abig, S& asml)
{
    // four independent loads per iteration, then the remaining rows
    int64_t r = r0;
    for(; r + 3 * BX < re; r += 4 * BX)
    {
        const T x0 = ac[r];
        const T x1 = ac[r + BX];
        const T x2 = ac[r + 2 * BX];
        const T x3 = ac[r + 3 * BX];
        lange_col_add<NORM, T, S>(x0, acc, abig, asml);
        lange_col_add<NORM, T, S>(x1, acc, abig, asml);
        lange_col_add<NORM, T, S>(x2, acc, abig, asml);
        lange_col_add<NORM, T, S>(x3, acc, abig, asml);
    }
    for(; r < re; r += BX)
        lange_col_add<NORM, T, S>(ac[r], acc, abig, asml);
}

/** Sum, maximum or scaled sums of squares of the elements of the columns, by chunks of rows. Grid
    (gx, nrch, batch_count), block (BX, BY) with BY = LANGE_THDS / BX: thread (tx, ty) visits rows
    tx, tx + BX, ... of the columns cb * BY + ty within the chunk blockIdx.y, for cb = blockIdx.x,
    blockIdx.x + gx, ... For the 1-norm, the sum of each column is computed; with several chunks
    of rows (nrch > 1), the result for column c and chunk h is part[(b * nrch + h) * n + c], and
    otherwise the maximum of the column sums of the block is part[b * gx + blockIdx.x]. For the
    other norms, each thread accumulates all its elements, and the maximum or the sum of squares
    of the block is part[(b * nrch + h) * gx + blockIdx.x] (for the Frobenius norm, the three
    accumulators are in part, part + psize and part + 2 * psize). **/
template <int NORM, int BX, typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(LANGE_THDS) lange_cols_kernel(const I m,
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
    constexpr int BY = LANGE_THDS / BX;
    const I b = blockIdx.z;
    const I h = blockIdx.y;
    const I tx = threadIdx.x;
    const I ty = threadIdx.y;
    const int64_t ncb = (int64_t(n) - 1) / BY + 1;
    const bool percol = (NORM == LANGE_NORM_SUM && nrch > 1);

    T* a = load_ptr_batch<T>(A, b, shiftA, strideA);

    // the medium accumulator (or the sum or the maximum), then the big and small accumulators
    __shared__ S sacc[3][LANGE_THDS];
    const int t = ty * BX + tx;

    // the indices are computed in 64 bits, as they can exceed the range of I near its end
    const int64_t rs = int64_t(h) * rchunk;
    const int64_t re = std::min<int64_t>(rs + rchunk, m);

    // the accumulators of the thread (for the 1-norm, the maximum of the column sums, kept by the
    // threads with tx = 0)
    S cacc = 0, cbig = 0, csml = 0;

    for(int64_t cb = blockIdx.x; cb < ncb; cb += gridDim.x)
    {
        const int64_t c = cb * BY + ty;

        if constexpr(NORM == LANGE_NORM_SUM)
        {
            // (big and sml are not used by the 1-norm)
            S sum = 0, big = 0, sml = 0;
            if(c < n)
                lange_col_rows<NORM, BX, T, S>(a + idx2D(0, c, lda), rs + tx, re, sum, big, sml);

            // reduce over tx (in a fixed order)
            if constexpr(BX > 1)
            {
                sacc[0][t] = sum;
                __syncthreads();
                for(int k = BX / 2; k > 0; k /= 2)
                {
                    if(tx < k)
                        sacc[0][t] += sacc[0][t + k];
                    __syncthreads();
                }
                if(tx == 0)
                    sum = sacc[0][t];
                __syncthreads();
            }

            if(tx == 0 && c < n)
            {
                if(percol)
                    part[(rocblas_stride(b) * nrch + h) * n + c] = sum;
                else
                    cacc = rocblas_max_nan(cacc, sum);
            }
        }
        else if(c < n)
        {
            lange_col_rows<NORM, BX, T, S>(a + idx2D(0, c, lda), rs + tx, re, cacc, cbig, csml);

            // keep the medium accumulator below tmed (see lange_ssq_merge)
            if constexpr(NORM == LANGE_NORM_FRO)
                lange_ssq_renorm<S>(cbig, cacc);
        }
    }

    // combine the results of the threads of the block (in a fixed order)
    if(!percol)
    {
        sacc[0][t] = cacc;
        sacc[1][t] = cbig;
        sacc[2][t] = csml;
        __syncthreads();
        for(int k = LANGE_THDS / 2; k > 0; k /= 2)
        {
            if(t < k)
                lange_cols_combine<NORM, S>(sacc[0][t], sacc[1][t], sacc[2][t], sacc[0][t + k],
                                            sacc[1][t + k], sacc[2][t + k]);
            __syncthreads();
        }

        if(t == 0)
        {
            const rocblas_stride p = (rocblas_stride(b) * nrch + h) * gridDim.x + blockIdx.x;
            part[p] = sacc[0][0];
            if constexpr(NORM == LANGE_NORM_FRO)
            {
                part[p + psize] = sacc[1][0];
                part[p + 2 * psize] = sacc[2][0];
            }
        }
    }
}

/** Sums of the absolute values of the elements of each row, by chunks of columns. Grid
    (gx, ncch, batch_count), block (BX, BY) with BY = LANGE_THDS / BX: thread (tx, ty) takes the
    rows rb * BX + tx, for rb = blockIdx.x, blockIdx.x + gx, ..., and the columns ty, ty + BY, ...
    of the chunk blockIdx.y. With several chunks of columns (ncch > 1), the result for row r and
    chunk h is part[(b * ncch + h) * m + r]; otherwise the maximum of the row sums of the block is
    part[b * gx + blockIdx.x]. **/
template <int BX, typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(LANGE_THDS) lange_rows_kernel(const I m,
                                                                      const I n,
                                                                      U A,
                                                                      const rocblas_stride shiftA,
                                                                      const I lda,
                                                                      const rocblas_stride strideA,
                                                                      const I cchunk,
                                                                      const I ncch,
                                                                      S* part)
{
    constexpr int BY = LANGE_THDS / BX;
    const I b = blockIdx.z;
    const I h = blockIdx.y;
    const I tx = threadIdx.x;
    const I ty = threadIdx.y;
    const int64_t nrb = (int64_t(m) - 1) / BX + 1;

    T* a = load_ptr_batch<T>(A, b, shiftA, strideA);

    __shared__ S ssum[BY][BX];

    // the indices are computed in 64 bits, as they can exceed the range of I near its end
    const int64_t cs = int64_t(h) * cchunk;
    const int64_t ce = std::min<int64_t>(cs + cchunk, n);

    // the maximum of the row sums of thread (tx, 0)
    S rmax = 0;

    for(int64_t rb = blockIdx.x; rb < nrb; rb += gridDim.x)
    {
        const int64_t r = rb * BX + tx;

        // four columns per iteration, then the remaining columns
        S sum = 0;
        if(r < m)
        {
            int64_t c = cs + ty;
            for(; c + 3 * BY < ce; c += 4 * BY)
            {
                sum += lange_abs(a[idx2D(r, c, lda)]);
                sum += lange_abs(a[idx2D(r, c + BY, lda)]);
                sum += lange_abs(a[idx2D(r, c + 2 * BY, lda)]);
                sum += lange_abs(a[idx2D(r, c + 3 * BY, lda)]);
            }
            for(; c < ce; c += BY)
                sum += lange_abs(a[idx2D(r, c, lda)]);
        }

        // reduce over ty (in a fixed order)
        ssum[ty][tx] = sum;
        __syncthreads();
        for(int k = BY / 2; k > 0; k /= 2)
        {
            if(ty < k)
                ssum[ty][tx] += ssum[ty + k][tx];
            __syncthreads();
        }
        if(ty == 0 && r < m)
        {
            if(ncch > 1)
                part[(rocblas_stride(b) * ncch + h) * m + r] = ssum[0][tx];
            else
                rmax = rocblas_max_nan(rmax, ssum[0][tx]);
        }
        __syncthreads();
    }

    // maximum of the row sums of the block (in a fixed order)
    if(ncch == 1)
    {
        if(ty == 0)
            ssum[0][tx] = rmax;
        __syncthreads();
        for(int k = BX / 2; k > 0; k /= 2)
        {
            if(ty == 0 && tx < k)
                ssum[0][tx] = rocblas_max_nan(ssum[0][tx], ssum[0][tx + k]);
            __syncthreads();
        }
        if(tx == 0 && ty == 0)
            part[rocblas_stride(b) * gridDim.x + blockIdx.x] = ssum[0][0];
    }
}

/** Combines the partial results part[(b * nch + h) * len + k] (h < nch, k < len) into the norm of
    each matrix: the maximum over k of the sums over h (1- and infinity-norms), or the maximum or
    the scaled sum of squares of all of them. Grid (1, 1, batch_count), block LANGE_FINAL_THDS. **/
template <int NORM, typename I, typename S>
ROCSOLVER_KERNEL void __launch_bounds__(LANGE_FINAL_THDS)
    lange_final_kernel(const I len, const I nch, const S* part, const rocblas_stride psize, S* norms)
{
    const I b = blockIdx.z;
    const I tid = threadIdx.x;

    __shared__ S sacc[3][LANGE_FINAL_THDS];

    S acc = 0, abig = 0, asml = 0;
    const S* p = part + rocblas_stride(b) * nch * len;

    if constexpr(NORM == LANGE_NORM_SUM)
    {
        for(int64_t k = tid; k < len; k += LANGE_FINAL_THDS)
        {
            S sum = 0;
            for(I h = 0; h < nch; h++)
                sum += p[rocblas_stride(h) * len + k];
            acc = rocblas_max_nan(acc, sum);
        }
    }
    else
    {
        const rocblas_stride total = rocblas_stride(nch) * len;
        for(rocblas_stride k = tid; k < total; k += LANGE_FINAL_THDS)
        {
            if constexpr(NORM == LANGE_NORM_MAX)
                acc = rocblas_max_nan(acc, p[k]);
            else
                lange_ssq_merge<S>(abig, acc, asml, p[k + psize], p[k], p[k + 2 * psize]);
        }
    }

    sacc[0][tid] = acc;
    if constexpr(NORM == LANGE_NORM_FRO)
    {
        sacc[1][tid] = abig;
        sacc[2][tid] = asml;
    }
    __syncthreads();
    for(I k = LANGE_FINAL_THDS / 2; k > 0; k /= 2)
    {
        if(tid < k)
        {
            if constexpr(NORM == LANGE_NORM_FRO)
                lange_ssq_merge<S>(sacc[1][tid], sacc[0][tid], sacc[2][tid], sacc[1][tid + k],
                                   sacc[0][tid + k], sacc[2][tid + k]);
            else
                sacc[0][tid] = rocblas_max_nan(sacc[0][tid], sacc[0][tid + k]);
        }
        __syncthreads();
    }

    if(tid == 0)
    {
        if constexpr(NORM == LANGE_NORM_FRO)
            norms[b] = lange_ssq_combine<S>(sacc[1][0], sacc[0][0], sacc[2][0]);
        else
            norms[b] = sacc[0][0];
    }
}

/** Chunk sizes and numbers of chunks of the column and row kernels, and their numbers of blocks of
    columns or rows (the sizes of the first dimension of their grids). **/
template <typename I>
void rocsolver_lange_chunks(const I m, const I n, I* rchunk, I* nrch, I* ncb, I* cchunk, I* ncch, I* nrb)
{
    const int64_t bx = lange_bx(m, n);
    const int64_t by = LANGE_THDS / bx;

    // column kernel: ceil(n / by) blocks of columns per chunk of rows, at most LANGE_MAX_BLOCKS of
    // them in the grid
    const int64_t nc = (int64_t(n) - 1) / by + 1;
    int64_t nch
        = std::min((LANGE_TARGET_BLOCKS - 1) / nc + 1, (int64_t(m) - 1) / LANGE_MIN_CHUNK + 1);
    int64_t len = ((int64_t(m) - 1) / nch / LANGE_BX + 1) * LANGE_BX;
    *rchunk = I(len);
    *nrch = I((int64_t(m) - 1) / len + 1);
    *ncb = I(std::min<int64_t>(nc, LANGE_MAX_BLOCKS));

    // row kernel: ceil(m / bx) blocks of rows per chunk of columns, at most LANGE_MAX_BLOCKS of
    // them in the grid
    const int64_t nr = (int64_t(m) - 1) / bx + 1;
    nch = std::min((LANGE_TARGET_BLOCKS - 1) / nr + 1, (int64_t(n) - 1) / LANGE_BX + 1);
    len = ((int64_t(n) - 1) / nch / LANGE_BX + 1) * LANGE_BX;
    *cchunk = I(len);
    *ncch = I((int64_t(n) - 1) / len + 1);
    *nrb = I(std::min<int64_t>(nr, LANGE_MAX_BLOCKS));
}

template <typename T, typename I, typename S>
void rocsolver_lange_getMemorySize(rocblas_handle handle,
                                   const rocsolver_norm_type norm_type,
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

    I rchunk, nrch, ncb, cchunk, ncch, nrb;
    rocsolver_lange_chunks(m, n, &rchunk, &nrch, &ncb, &cchunk, &ncch, &nrb);

    // partial results of the row kernel (infinity-norm; per row with several chunks of columns,
    // per block otherwise) or of the column kernel (per column for the 1-norm with several chunks
    // of rows, per block otherwise, and three accumulators for the Frobenius norm)
    size_t size;
    if(norm_type == rocsolver_norm_type_infinity)
        size = (ncch > 1) ? size_t(ncch) * m : size_t(nrb);
    else if(norm_type == rocsolver_norm_type_one && nrch > 1)
        size = size_t(nrch) * n;
    else
        size = (norm_type == rocsolver_norm_type_frobenius ? 3 : 1) * size_t(nrch) * ncb;
    *size_work = sizeof(S) * size * batch_count;
}

/** Launches lange_cols_kernel with bx threads per column. **/
template <int NORM, typename T, typename I, typename S, typename U>
void rocsolver_lange_cols(rocblas_handle handle,
                          hipStream_t stream,
                          const int bx,
                          const dim3 grid,
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
    const dim3 threads(bx, LANGE_THDS / bx, 1);
    switch(bx)
    {
    case 1:
        ROCSOLVER_LAUNCH_KERNEL((lange_cols_kernel<NORM, 1, T>), grid, threads, 0, stream, m, n, A,
                                shiftA, lda, strideA, rchunk, nrch, part, psize);
        break;
    case 4:
        ROCSOLVER_LAUNCH_KERNEL((lange_cols_kernel<NORM, 4, T>), grid, threads, 0, stream, m, n, A,
                                shiftA, lda, strideA, rchunk, nrch, part, psize);
        break;
    case 16:
        ROCSOLVER_LAUNCH_KERNEL((lange_cols_kernel<NORM, 16, T>), grid, threads, 0, stream, m, n, A,
                                shiftA, lda, strideA, rchunk, nrch, part, psize);
        break;
    case LANGE_THDS:
        ROCSOLVER_LAUNCH_KERNEL((lange_cols_kernel<NORM, LANGE_THDS, T>), grid, threads, 0, stream,
                                m, n, A, shiftA, lda, strideA, rchunk, nrch, part, psize);
        break;
    default:
        ROCSOLVER_LAUNCH_KERNEL((lange_cols_kernel<NORM, LANGE_BX, T>), grid, threads, 0, stream, m,
                                n, A, shiftA, lda, strideA, rchunk, nrch, part, psize);
    }
}

/** Launches lange_rows_kernel with bx rows per block. **/
template <typename T, typename I, typename S, typename U>
void rocsolver_lange_rows(rocblas_handle handle,
                          hipStream_t stream,
                          const int bx,
                          const dim3 grid,
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
    const dim3 threads(bx, LANGE_THDS / bx, 1);
    switch(bx)
    {
    case 1:
        ROCSOLVER_LAUNCH_KERNEL((lange_rows_kernel<1, T>), grid, threads, 0, stream, m, n, A,
                                shiftA, lda, strideA, cchunk, ncch, part);
        break;
    case 4:
        ROCSOLVER_LAUNCH_KERNEL((lange_rows_kernel<4, T>), grid, threads, 0, stream, m, n, A,
                                shiftA, lda, strideA, cchunk, ncch, part);
        break;
    case 16:
        ROCSOLVER_LAUNCH_KERNEL((lange_rows_kernel<16, T>), grid, threads, 0, stream, m, n, A,
                                shiftA, lda, strideA, cchunk, ncch, part);
        break;
    case LANGE_THDS:
        ROCSOLVER_LAUNCH_KERNEL((lange_rows_kernel<LANGE_THDS, T>), grid, threads, 0, stream, m, n,
                                A, shiftA, lda, strideA, cchunk, ncch, part);
        break;
    default:
        ROCSOLVER_LAUNCH_KERNEL((lange_rows_kernel<LANGE_BX, T>), grid, threads, 0, stream, m, n, A,
                                shiftA, lda, strideA, cchunk, ncch, part);
    }
}

template <typename T, typename I, typename S>
rocblas_status rocsolver_lange_argCheck(rocblas_handle handle,
                                        const rocsolver_norm_type norm_type,
                                        const I m,
                                        const I n,
                                        const I lda,
                                        T A,
                                        S* norms)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(norm_type != rocsolver_norm_type_one && norm_type != rocsolver_norm_type_frobenius
       && norm_type != rocsolver_norm_type_infinity && norm_type != rocsolver_norm_type_max)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(m < 0 || n < 0 || lda < m)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((m && n && !A) || (m && n && !norms))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename I, typename S, typename U>
rocblas_status rocsolver_lange_template(rocblas_handle handle,
                                        const rocsolver_norm_type norm_type,
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
    ROCSOLVER_ENTER("lange", "norm_type:", norm_type, "m:", m, "n:", n, "shiftA:", shiftA,
                    "lda:", lda, "bc:", batch_count);

    // quick return
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

    const int bx = lange_bx(m, n);
    I rchunk, nrch, ncb, cchunk, ncch, nrb;
    rocsolver_lange_chunks(m, n, &rchunk, &nrch, &ncb, &cchunk, &ncch, &nrb);

    // the partial results of the column kernel are per column for the 1-norm with several chunks
    // of rows, and per block otherwise; for the Frobenius norm, its three accumulators are in three
    // arrays of psize elements
    const bool percol = (norm_type == rocsolver_norm_type_one && nrch > 1);
    const I clen = percol ? n : ncb;
    const rocblas_stride psize = rocblas_stride(nrch) * ncb * batch_count;

    // the partial results of the row kernel are per row with several chunks of columns, and per
    // block otherwise
    const I rlen = (ncch > 1) ? m : nrb;

    dim3 gridc(ncb, nrch, batch_count);
    dim3 gridr(nrb, ncch, batch_count);
    dim3 gridf(1, 1, batch_count);
    dim3 threadsf(LANGE_FINAL_THDS);

    switch(norm_type)
    {
    case rocsolver_norm_type_one:
        rocsolver_lange_cols<LANGE_NORM_SUM, T>(handle, stream, bx, gridc, m, n, A, shiftA, lda,
                                                strideA, rchunk, nrch, work, psize);
        ROCSOLVER_LAUNCH_KERNEL((lange_final_kernel<LANGE_NORM_SUM, I, S>), gridf, threadsf, 0,
                                stream, clen, nrch, work, psize, norms);
        break;
    case rocsolver_norm_type_infinity:
        rocsolver_lange_rows<T>(handle, stream, bx, gridr, m, n, A, shiftA, lda, strideA, cchunk,
                                ncch, work);
        ROCSOLVER_LAUNCH_KERNEL((lange_final_kernel<LANGE_NORM_SUM, I, S>), gridf, threadsf, 0,
                                stream, rlen, ncch, work, psize, norms);
        break;
    case rocsolver_norm_type_max:
        rocsolver_lange_cols<LANGE_NORM_MAX, T>(handle, stream, bx, gridc, m, n, A, shiftA, lda,
                                                strideA, rchunk, nrch, work, psize);
        ROCSOLVER_LAUNCH_KERNEL((lange_final_kernel<LANGE_NORM_MAX, I, S>), gridf, threadsf, 0,
                                stream, clen, nrch, work, psize, norms);
        break;
    case rocsolver_norm_type_frobenius:
        rocsolver_lange_cols<LANGE_NORM_FRO, T>(handle, stream, bx, gridc, m, n, A, shiftA, lda,
                                                strideA, rchunk, nrch, work, psize);
        ROCSOLVER_LAUNCH_KERNEL((lange_final_kernel<LANGE_NORM_FRO, I, S>), gridf, threadsf, 0,
                                stream, clen, nrch, work, psize, norms);
        break;
    default: return rocblas_status_invalid_value;
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE

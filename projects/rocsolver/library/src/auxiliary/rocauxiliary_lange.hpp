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
 * of LANGE_BX threads goes down one column, so the reads are coalesced), and the row sums of the
 * infinity-norm by lange_rows_kernel (each thread takes one row and the threads of a block read
 * the same columns, so these reads are coalesced too). The columns (rows) are split into chunks
 * so that the grid has about LANGE_TARGET_BLOCKS blocks; each chunk writes a partial result, and
 * lange_final_kernel adds the partial results in a fixed order (the result does not depend on the
 * scheduling of the blocks).
 *
 * The Frobenius norm uses the scaled sums of squares of LAPACK's xLASSQ (Blue's algorithm): each
 * element goes into one of three accumulators (big, medium, small) depending on its magnitude, so
 * that the sum of squares does not overflow or underflow unless the norm itself does. As in
 * xLASSQ, the real and imaginary parts of complex elements are taken separately.
 *************************************************************/

#define LANGE_NORM_SUM 0
#define LANGE_NORM_MAX 1
#define LANGE_NORM_FRO 2

#ifndef LANGE_BX
#define LANGE_BX 64
#endif
#ifndef LANGE_BY
#define LANGE_BY 4
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
};

template <>
struct lange_blue<double>
{
    static constexpr double tsml = 0x1p-511;
    static constexpr double tbig = 0x1p486;
    static constexpr double ssml = 0x1p537;
    static constexpr double sbig = 0x1p-538;
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

/** Sum, maximum or scaled sums of squares of the elements of each column, by chunks of rows. Grid
    (ceil(n / LANGE_BY), nrch, batch_count), block (LANGE_BX, LANGE_BY): thread (tx, ty) visits
    rows tx, tx + LANGE_BX, ... of column blockIdx.x * LANGE_BY + ty within the chunk blockIdx.y.
    The result for column c and chunk h is part[(b * nrch + h) * n + c] (and, for the Frobenius
    norm, the three accumulators are in part, part + psize and part + 2 * psize). **/
template <int NORM, typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(LANGE_BX* LANGE_BY)
    lange_cols_kernel(const I m,
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
    const I ncb = (n - 1) / LANGE_BY + 1;

    T* a = load_ptr_batch<T>(A, b, shiftA, strideA);

    // the medium accumulator (or the sum or the maximum), then the big and small accumulators
    __shared__ S sacc[3][LANGE_BY][LANGE_BX];

    // the indices are computed in 64 bits, as they can exceed the range of I near its end
    const int64_t rs = int64_t(h) * rchunk;
    const int64_t re = std::min<int64_t>(rs + rchunk, m);

    for(int64_t cb = blockIdx.x; cb < ncb; cb += gridDim.x)
    {
        const int64_t c = cb * LANGE_BY + ty;
        S acc = 0, abig = 0, asml = 0;

        if(c < n)
        {
            // four independent loads per iteration, then the remaining rows
            const T* ac = a + idx2D(0, c, lda);
            int64_t r = rs + tx;
            for(; r + 3 * LANGE_BX < re; r += 4 * LANGE_BX)
            {
                const T x0 = ac[r];
                const T x1 = ac[r + LANGE_BX];
                const T x2 = ac[r + 2 * LANGE_BX];
                const T x3 = ac[r + 3 * LANGE_BX];
                lange_col_add<NORM, T, S>(x0, acc, abig, asml);
                lange_col_add<NORM, T, S>(x1, acc, abig, asml);
                lange_col_add<NORM, T, S>(x2, acc, abig, asml);
                lange_col_add<NORM, T, S>(x3, acc, abig, asml);
            }
            for(; r < re; r += LANGE_BX)
                lange_col_add<NORM, T, S>(ac[r], acc, abig, asml);
        }

        // reduce over tx (in a fixed order)
        sacc[0][ty][tx] = acc;
        if constexpr(NORM == LANGE_NORM_FRO)
        {
            sacc[1][ty][tx] = abig;
            sacc[2][ty][tx] = asml;
        }
        __syncthreads();
        for(I k = LANGE_BX / 2; k > 0; k /= 2)
        {
            if(tx < k)
            {
                if constexpr(NORM == LANGE_NORM_MAX)
                    sacc[0][ty][tx] = rocblas_max_nan(sacc[0][ty][tx], sacc[0][ty][tx + k]);
                else
                    sacc[0][ty][tx] += sacc[0][ty][tx + k];
                if constexpr(NORM == LANGE_NORM_FRO)
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
            if constexpr(NORM == LANGE_NORM_FRO)
            {
                part[p + psize] = sacc[1][ty][0];
                part[p + 2 * psize] = sacc[2][ty][0];
            }
        }
        __syncthreads();
    }
}

/** Sums of the absolute values of the elements of each row, by chunks of columns. Grid
    (ceil(m / LANGE_BX), ncch, batch_count), block (LANGE_BX, LANGE_BY): thread (tx, ty) takes row
    blockIdx.x * LANGE_BX + tx and the columns ty, ty + LANGE_BY, ... of the chunk blockIdx.y. The
    result for row r and chunk h is part[(b * ncch + h) * m + r]. **/
template <typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(LANGE_BX* LANGE_BY)
    lange_rows_kernel(const I m,
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
    const I nrb = (m - 1) / LANGE_BX + 1;

    T* a = load_ptr_batch<T>(A, b, shiftA, strideA);

    __shared__ S ssum[LANGE_BY][LANGE_BX];

    // the indices are computed in 64 bits, as they can exceed the range of I near its end
    const int64_t cs = int64_t(h) * cchunk;
    const int64_t ce = std::min<int64_t>(cs + cchunk, n);

    for(int64_t rb = blockIdx.x; rb < nrb; rb += gridDim.x)
    {
        const int64_t r = rb * LANGE_BX + tx;

        // four columns per iteration, then the remaining columns
        S sum = 0;
        if(r < m)
        {
            int64_t c = cs + ty;
            for(; c + 3 * LANGE_BY < ce; c += 4 * LANGE_BY)
            {
                sum += lange_abs(a[idx2D(r, c, lda)]);
                sum += lange_abs(a[idx2D(r, c + LANGE_BY, lda)]);
                sum += lange_abs(a[idx2D(r, c + 2 * LANGE_BY, lda)]);
                sum += lange_abs(a[idx2D(r, c + 3 * LANGE_BY, lda)]);
            }
            for(; c < ce; c += LANGE_BY)
                sum += lange_abs(a[idx2D(r, c, lda)]);
        }

        ssum[ty][tx] = sum;
        __syncthreads();
        if(ty == 0 && r < m)
        {
            for(I k = 1; k < LANGE_BY; k++)
                sum += ssum[k][tx];
            part[(rocblas_stride(b) * ncch + h) * m + r] = sum;
        }
        __syncthreads();
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
            {
                acc += p[k];
                abig += p[k + psize];
                asml += p[k + 2 * psize];
            }
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
        if constexpr(NORM == LANGE_NORM_FRO)
            norms[b] = lange_ssq_combine<S>(sacc[1][0], sacc[0][0], sacc[2][0]);
        else
            norms[b] = sacc[0][0];
    }
}

/** Chunk sizes and numbers of chunks of the column and row kernels. **/
template <typename I>
void rocsolver_lange_chunks(const I m, const I n, I* rchunk, I* nrch, I* cchunk, I* ncch)
{
    // column kernel: ceil(n / LANGE_BY) blocks per chunk of rows
    const int64_t ncb = (int64_t(n) - 1) / LANGE_BY + 1;
    int64_t nch
        = std::min((LANGE_TARGET_BLOCKS - 1) / ncb + 1, (int64_t(m) - 1) / LANGE_MIN_CHUNK + 1);
    int64_t len = ((int64_t(m) - 1) / nch / LANGE_BX + 1) * LANGE_BX;
    *rchunk = I(len);
    *nrch = I((int64_t(m) - 1) / len + 1);

    // row kernel: ceil(m / LANGE_BX) blocks per chunk of columns
    const int64_t nrb = (int64_t(m) - 1) / LANGE_BX + 1;
    nch = std::min((LANGE_TARGET_BLOCKS - 1) / nrb + 1, (int64_t(n) - 1) / LANGE_BX + 1);
    len = ((int64_t(n) - 1) / nch / LANGE_BX + 1) * LANGE_BX;
    *cchunk = I(len);
    *ncch = I((int64_t(n) - 1) / len + 1);
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

    I rchunk, nrch, cchunk, ncch;
    rocsolver_lange_chunks(m, n, &rchunk, &nrch, &cchunk, &ncch);

    // partial results of the row kernel (infinity-norm) or of the column kernel (three
    // accumulators for the Frobenius norm)
    size_t size;
    if(norm_type == rocsolver_norm_type_infinity)
        size = size_t(ncch) * m;
    else
        size = (norm_type == rocsolver_norm_type_frobenius ? 3 : 1) * size_t(nrch) * n;
    *size_work = sizeof(S) * size * batch_count;
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

    const hipDeviceProp_t* props = rocblas_internal_get_device_prop(handle);
    const int64_t maxgrid = props->maxGridSize[0];

    I rchunk, nrch, cchunk, ncch;
    rocsolver_lange_chunks(m, n, &rchunk, &nrch, &cchunk, &ncch);

    // for the Frobenius norm, the three accumulators of the column kernel are in three arrays of
    // psize elements
    const rocblas_stride psize = rocblas_stride(nrch) * n * batch_count;

    dim3 threads(LANGE_BX, LANGE_BY, 1);
    // the kernels loop over the blocks beyond the grid; the number of threads in each dimension
    // of the grid must be less than 2^32
    const int64_t maxblocks = std::min<int64_t>(maxgrid, ((int64_t(1) << 32) - 1) / LANGE_BX);
    const I ncb = I(std::min<int64_t>((int64_t(n) - 1) / LANGE_BY + 1, maxblocks));
    const I nrb = I(std::min<int64_t>((int64_t(m) - 1) / LANGE_BX + 1, maxblocks));
    dim3 gridc(ncb, nrch, batch_count);
    dim3 gridr(nrb, ncch, batch_count);
    dim3 gridf(1, 1, batch_count);
    dim3 threadsf(LANGE_FINAL_THDS);

    switch(norm_type)
    {
    case rocsolver_norm_type_one:
        ROCSOLVER_LAUNCH_KERNEL((lange_cols_kernel<LANGE_NORM_SUM, T>), gridc, threads, 0, stream,
                                m, n, A, shiftA, lda, strideA, rchunk, nrch, work, psize);
        ROCSOLVER_LAUNCH_KERNEL((lange_final_kernel<LANGE_NORM_SUM, I, S>), gridf, threadsf, 0,
                                stream, n, nrch, work, psize, norms);
        break;
    case rocsolver_norm_type_infinity:
        ROCSOLVER_LAUNCH_KERNEL((lange_rows_kernel<T>), gridr, threads, 0, stream, m, n, A, shiftA,
                                lda, strideA, cchunk, ncch, work);
        ROCSOLVER_LAUNCH_KERNEL((lange_final_kernel<LANGE_NORM_SUM, I, S>), gridf, threadsf, 0,
                                stream, m, ncch, work, psize, norms);
        break;
    case rocsolver_norm_type_max:
        ROCSOLVER_LAUNCH_KERNEL((lange_cols_kernel<LANGE_NORM_MAX, T>), gridc, threads, 0, stream,
                                m, n, A, shiftA, lda, strideA, rchunk, nrch, work, psize);
        ROCSOLVER_LAUNCH_KERNEL((lange_final_kernel<LANGE_NORM_MAX, I, S>), gridf, threadsf, 0,
                                stream, n, nrch, work, psize, norms);
        break;
    case rocsolver_norm_type_frobenius:
        ROCSOLVER_LAUNCH_KERNEL((lange_cols_kernel<LANGE_NORM_FRO, T>), gridc, threads, 0, stream,
                                m, n, A, shiftA, lda, strideA, rchunk, nrch, work, psize);
        ROCSOLVER_LAUNCH_KERNEL((lange_final_kernel<LANGE_NORM_FRO, I, S>), gridf, threadsf, 0,
                                stream, n, nrch, work, psize, norms);
        break;
    default: return rocblas_status_invalid_value;
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE

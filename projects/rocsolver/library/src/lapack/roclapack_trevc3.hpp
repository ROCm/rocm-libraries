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

/*
 * ===========================================================================
 *    TREVC3 computes the eigenvectors of an upper triangular matrix T as in
 *    LAPACK ZTREVC3, for blocks of TREVC3_NC eigenvectors at a time. The
 *    triangular systems (T - lambda_k I) x = 0 (right) or
 *    (T - lambda_k I)^H y = 0 (left) of a block are solved by substitution over
 *    TREVC3_NB-row diagonal blocks of T: the coupling with the rows already
 *    solved is a matrix-matrix product, and the shifted solves with the diagonal
 *    block run with one thread per eigenvector (the diagonal block in shared
 *    memory). As in ZTREVC3, a pivot smaller than
 *    smin = max(ulp*|lambda_k|, smlnum) is replaced by smin. Overflow is
 *    controlled by scaling each vector (the role of ZLATRS): the entries are
 *    kept below xbig = hugeval / 64 / max(1, tmax), where tmax bounds the row and
 *    column sums of the strictly upper triangular part of T, so that no product
 *    or sum overflows (tmax is computed scaled by 1/(4n), as it can overflow for
 *    finite T); any positive scaling is valid as the vectors are normalized at
 *    the end. The back-transformation (Q x) is a matrix-matrix product, in place
 *    in VR (VL), with the blocks processed in the order in which the columns of Q
 *    that are still needed are never overwritten; as in the blocked ZTREVC3, each
 *    vector is first scaled to max |Re| + |Im| = 1, since Q x can overflow for x
 *    near xbig.
 * ===========================================================================
 */

/** TREVC3_BLOCK_MAX returns the maximum of v over the threads of the block
    (all the threads must call it). **/
template <int BS, typename S>
__device__ S trevc3_block_max(S v, S* sred)
{
    const int tid = hipThreadIdx_x;
    sred[tid] = v;
    __syncthreads();
    for(int s = BS / 2; s > 0; s /= 2)
    {
        if(tid < s)
            sred[tid] = std::max(sred[tid], sred[tid + s]);
        __syncthreads();
    }
    const S r = sred[0];
    __syncthreads();
    return r;
}

/** TREVC3_NORMS computes, for each matrix of the batch, tmax/(4n), where tmax is the
    largest row or column sum (with |Re| + |Im|) of the strictly upper triangular part of T.
    The scaled sums cannot overflow for finite T (tmax itself, or even |Re| + |Im| of one
    entry, can): each of the at most n-1 scaled terms of a sum is at most huge/(2n), so the
    exact sum is below huge/2, which leaves a factor 2 for the rounding errors of the
    summation (up to about n*ulp relative; a scaling by 1/(2n) would leave only a factor
    n/(n-1), not enough for n in the thousands in single precision).
    The column sums (trevc3_colsums_kernel, a wavefront per column) and the row sums
    (trevc3_rowsums_kernel, a thread-block per 64 rows) are computed by many thread-blocks,
    each sum in a fixed order; their maximum is taken with atomic operations on the bits of
    the (non-negative) sums, whose order does not change it, so the result is deterministic.
    Sums that are NaN are ignored. **/
template <typename S>
__device__ void trevc3_atomic_max(S* p, const S v)
{
    if constexpr(sizeof(S) == 8)
        atomicMax(reinterpret_cast<unsigned long long*>(p),
                  static_cast<unsigned long long>(__double_as_longlong(v)));
    else
        atomicMax(reinterpret_cast<unsigned int*>(p), __float_as_uint(v));
}

template <int BS, typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(BS) trevc3_colsums_kernel(const I n,
                                                                  U TT,
                                                                  const rocblas_stride shiftT,
                                                                  const I ldt,
                                                                  const rocblas_stride strideT,
                                                                  S* tmax_s)
{
    const I bid = hipBlockIdx_y;
    const T* A = load_ptr_batch<T>(TT, bid, shiftT, strideT);
    const S c = S(0.25) / S(n);
    const int lane = hipThreadIdx_x % warpSize;
    const int nwaves = BS / warpSize;
    const I wave = hipBlockIdx_x * nwaves + hipThreadIdx_x / warpSize;
    S m = 0;
    for(I j = wave; j < n; j += I(hipGridDim_x) * nwaves)
    {
        S cs = 0;
        for(I l = lane; l < j; l += warpSize)
        {
            const T t = A[l + j * size_t(ldt)];
            cs += std::abs(t.real()) * c + std::abs(t.imag()) * c;
        }
        for(int off = warpSize / 2; off > 0; off /= 2)
            cs += __shfl_down(cs, off);
        if(cs > m)
            m = cs;
    }
    if(lane == 0 && m > 0)
        trevc3_atomic_max(tmax_s + bid, m);
}

template <int BS, typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(BS) trevc3_rowsums_kernel(const I n,
                                                                  U TT,
                                                                  const rocblas_stride shiftT,
                                                                  const I ldt,
                                                                  const rocblas_stride strideT,
                                                                  S* tmax_s)
{
    // (64 rows per thread-block, and BS/64 threads per row, which take the columns in turn,
    // so that the threads of a wavefront read contiguous entries)
    constexpr int RB = 64;
    constexpr int NC = BS / RB;
    const I bid = hipBlockIdx_y;
    const T* A = load_ptr_batch<T>(TT, bid, shiftT, strideT);
    const S c = S(0.25) / S(n);
    const int r = hipThreadIdx_x % RB;
    const int cl = hipThreadIdx_x / RB;
    const I i0 = I(hipBlockIdx_x) * RB;
    const I i = i0 + r;
    S rs = 0;
    for(I l = i0 + 1 + cl; l < n; l += NC)
    {
        if(i < n && l > i)
        {
            const T t = A[i + l * size_t(ldt)];
            rs += std::abs(t.real()) * c + std::abs(t.imag()) * c;
        }
    }
    __shared__ S part[BS];
    part[hipThreadIdx_x] = rs;
    __syncthreads();
    if(cl == 0 && i < n)
    {
        for(int k = 1; k < NC; k++)
            rs += part[r + k * RB];
        if(rs > 0)
            trevc3_atomic_max(tmax_s + bid, rs);
    }
}

template <typename T, typename I, typename S, typename U>
rocblas_status trevc3_norms(rocblas_handle handle,
                            hipStream_t stream,
                            const I n,
                            U A,
                            const rocblas_stride shiftT,
                            const I ldt,
                            const rocblas_stride strideT,
                            const I batch_count,
                            S* tmax_s)
{
    constexpr int BS = 256;
    HIP_CHECK(hipMemsetAsync(tmax_s, 0, sizeof(S) * batch_count, stream));
    const I ncb = std::min((n - 1) / (BS / 64) + 1, I(4096));
    ROCSOLVER_LAUNCH_KERNEL((trevc3_colsums_kernel<BS, T>), dim3(ncb, batch_count), dim3(BS), 0,
                            stream, n, A, shiftT, ldt, strideT, tmax_s);
    ROCSOLVER_LAUNCH_KERNEL((trevc3_rowsums_kernel<BS, T>), dim3((n - 1) / 64 + 1, batch_count),
                            dim3(BS), 0, stream, n, A, shiftT, ldt, strideT, tmax_s);
    return rocblas_status_success;
}

/** TREVC3_XBIG returns the bound xbig = hugeval / 64 / max(1, tmax) on the entries of the
    vectors, from tmax_s = tmax/(4n) (see trevc3_norms), as
    (hugeval / (256n)) / max(1/(4n), tmax_s), without overflow (the quotient is at most
    hugeval/64) or harmful underflow (a tiny tmax_s loses to 1/(4n)). With it no product or
    sum in the solves or the coupling products overflows. **/
template <typename S, typename I>
__device__ S trevc3_xbig(const I n, const S tmax_s)
{
    const S c = S(0.25) / S(n);
    return (std::numeric_limits<S>::max() / (S(64) * S(4) * S(n))) / std::max(c, tmax_s);
}

/** TREVC3_SCALE_KERNEL scales the columns c = 0:nc-1 of X (leading dimension n) so that
    the entry of largest |Re| + |Im| in the rows r0(c):r1(c)-1 has |Re| + |Im| = 1, where
    r0 = 0 and r1 = p0+c+1 (right eigenvectors) or r0 = p0+c and r1 = n (left eigenvectors;
    the other rows are zero). Applied before the back-transformation Q*X, which would
    overflow for entries near xbig (the growth is up to about n*max|q_ij|). **/
template <int BS, typename T, typename I>
ROCSOLVER_KERNEL void __launch_bounds__(BS)
    trevc3_scale_kernel(const bool left, const I n, const I p0, T* XX, const rocblas_stride strideX)
{
    using S = decltype(std::real(T{}));

    const I c = hipBlockIdx_x;
    const I bid = hipBlockIdx_y;
    const I k = p0 + c;
    T* x = XX + bid * strideX + c * size_t(n);
    __shared__ S sred[BS];

    const I r0 = left ? k : 0;
    const I r1 = left ? n : k + 1;
    S m = 0;
    for(I r = r0 + hipThreadIdx_x; r < r1; r += BS)
        m = std::max(m, hqr_cabs1(x[r]));
    m = trevc3_block_max<BS>(m, sred);
    if(!(m > 0) || !std::isfinite(m))
        return;

    // (divided, as 1/m may overflow when m is tiny)
    for(I r = r0 + hipThreadIdx_x; r < r1; r += BS)
        x[r] = T(x[r].real() / m, x[r].imag() / m);
}

/** TREVC3_SOLVE_KERNEL solves the rows i0:i0+nb-1 (0-based; a diagonal block of T)
    of the eigenvectors k = p0:p1-1 that involve them, one thread per eigenvector.
    Right eigenvectors (LEFT = false, rows processed from the bottom): for k >= i0+nb,
    the right-hand side (the coupling with the rows below, already solved) is in R;
    for k in the block, x_k = 1 and the rows below it are zero. Left eigenvectors
    (LEFT = true, rows processed from the top): for k < i0, the right-hand side (the
    coupling with the rows above) is in R; for k in the block, y_k = 1 and the rows
    above it are zero. X (leading dimension n) holds the vectors of the block of
    eigenvectors; the rows solved before (outside this diagonal block) are rescaled
    if the vector is. **/
template <bool LEFT, int NB, int BS, typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(BS) trevc3_solve_kernel(const I n,
                                                                U TT,
                                                                const rocblas_stride shiftT,
                                                                const I ldt,
                                                                const rocblas_stride strideT,
                                                                const I i0,
                                                                const I nb,
                                                                const I p0,
                                                                const I p1,
                                                                const T* RR,
                                                                const rocblas_stride strideR,
                                                                T* XX,
                                                                const rocblas_stride strideX,
                                                                const S* tmax_s)
{
    const I bid = hipBlockIdx_y;
    const T* A = load_ptr_batch<T>(TT, bid, shiftT, strideT);
    const T* R = RR + bid * strideR;
    T* X = XX + bid * strideX;
    const I i1 = i0 + nb;

    // diagonal block of T in shared memory
    __shared__ T Ts[NB * NB];
    for(I e = hipThreadIdx_x; e < nb * nb; e += BS)
    {
        const I c = e / nb;
        const I r = e - c * nb;
        Ts[r + c * NB] = A[(i0 + r) + (i0 + c) * size_t(ldt)];
    }
    __syncthreads();

    // eigenvectors that involve these rows
    const I kbeg = LEFT ? p0 : std::max(i0, p0);
    const I kend = LEFT ? std::min(i1, p1) : p1;
    const I k = kbeg + hipBlockIdx_x * BS + hipThreadIdx_x;
    if(k >= kend)
        return;

    const T lam = A[k + k * size_t(ldt)];
    const S ulp = hqr_ulp<S>();
    const S smlnum = hqr_safmin<S>() * (S(n) / ulp);
    const S smin = std::max(ulp * hqr_cabs1(lam), smlnum);
    const S xbig = trevc3_xbig(n, tmax_s[bid]);
    const T* Rk = R + (k - p0) * NB;
    T* Xk = X + (k - p0) * size_t(n);

    T x[NB];
    I jfirst, jlast, lfirst, llast;
    if(k >= i0 && k < i1)
    {
        const I kl = k - i0;
        for(I j = 0; j < nb; j++)
            x[j] = T(0);
        x[kl] = T(std::min(S(1), xbig));
        jfirst = LEFT ? kl + 1 : kl - 1;
        lfirst = LEFT ? kl : 0;
        llast = LEFT ? nb - 1 : kl;
    }
    else
    {
        for(I j = 0; j < nb; j++)
            x[j] = Rk[j];
        jfirst = LEFT ? 0 : nb - 1;
        lfirst = 0;
        llast = nb - 1;
    }
    jlast = LEFT ? nb - 1 : 0;
    const I jstep = LEFT ? 1 : -1;

    // substitution, with the entries of x (solved or right-hand side) kept below xbig
    S stot = 1;
    for(I j = jfirst; LEFT ? j <= jlast : j >= jlast; j += jstep)
    {
        T s = x[j];
        T d;
        if(LEFT)
        {
            for(I l = lfirst; l < j; l++)
                s -= conj(Ts[l + j * NB]) * x[l];
            d = Ts[j + j * NB] - lam;
            if(hqr_cabs1(d) < smin)
                d = T(smin);
            d = conj(d);
        }
        else
        {
            for(I l = j + 1; l <= llast; l++)
                s -= Ts[j + l * NB] * x[l];
            d = Ts[j + j * NB] - lam;
            if(hqr_cabs1(d) < smin)
                d = T(smin);
        }

        const S as = hqr_cabs1(s);
        const S ad = hqr_cabs1(d);
        if(as > xbig * ad)
        {
            const S sc = (xbig * ad) / as;
            for(I l = lfirst; l <= llast; l++)
                x[l] *= sc;
            s *= sc;
            stot *= sc;
        }
        x[j] = hqr_zladiv(s, d);
    }

    // rescale the rows solved before, and store the new ones
    if(stot != S(1))
    {
        if(LEFT)
        {
            for(I r = std::max(k, p0); r < i0; r++)
                Xk[r] *= stot;
        }
        else
        {
            for(I r = i1; r <= k; r++)
                Xk[r] *= stot;
        }
    }
    for(I j = 0; j < nb; j++)
        Xk[i0 + j] = x[j];
}

/** TREVC3_NORMALIZE_KERNEL copies the columns c = 0:nc-1 of X (leading dimension n)
    into the columns p0+c of V, normalized so that the entry of largest |Re| + |Im|
    has |Re| + |Im| = 1. If all_right (all_left), only the rows 0:k (k:n-1), k = p0+c,
    are copied and the others are set to zero. **/
template <int BS, typename T, typename I, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(BS) trevc3_normalize_kernel(const bool all_right,
                                                                    const bool all_left,
                                                                    const I n,
                                                                    const I p0,
                                                                    const T* XX,
                                                                    const rocblas_stride strideX,
                                                                    U VV,
                                                                    const rocblas_stride shiftV,
                                                                    const I ldv,
                                                                    const rocblas_stride strideV)
{
    using S = decltype(std::real(T{}));

    const I c = hipBlockIdx_x;
    const I bid = hipBlockIdx_y;
    const I k = p0 + c;
    const T* x = XX + bid * strideX + c * size_t(n);
    T* v = load_ptr_batch<T>(VV, bid, shiftV, strideV) + k * size_t(ldv);
    __shared__ S sred[BS];

    const I r0 = all_left ? k : 0;
    const I r1 = all_right ? k + 1 : n;
    S m = 0;
    for(I r = r0 + hipThreadIdx_x; r < r1; r += BS)
        m = std::max(m, hqr_cabs1(x[r]));
    m = trevc3_block_max<BS>(m, sred);
    const S remax = (m > 0) ? S(1) / m : S(1);

    for(I r = hipThreadIdx_x; r < n; r += BS)
        v[r] = (r >= r0 && r < r1) ? x[r] * remax : T(0);
}

template <typename I, typename U>
rocblas_status rocsolver_trevc3_argCheck(rocblas_handle handle,
                                         const rocblas_side side,
                                         const rocsolver_eigenvectors howmny,
                                         const I n,
                                         const I ldt,
                                         const I ldvl,
                                         const I ldvr,
                                         U T,
                                         U VL,
                                         U VR,
                                         const I batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(side != rocblas_side_left && side != rocblas_side_right && side != rocblas_side_both)
        return rocblas_status_invalid_value;
    if(howmny != rocsolver_eigenvectors_all && howmny != rocsolver_eigenvectors_backtransform)
        return rocblas_status_invalid_value;
    const bool leftv = (side != rocblas_side_right);
    const bool rightv = (side != rocblas_side_left);

    // 2. invalid size
    if(n < 0 || ldt < n || ldt < 1 || ldvl < 1 || ldvr < 1 || (leftv && ldvl < n)
       || (rightv && ldvr < n) || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !T) || (n && leftv && !VL) || (n && rightv && !VR))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

// (side does not change the sizes; it is kept for symmetry with the other arguments)
template <bool BATCHED, typename T, typename I>
void rocsolver_trevc3_getMemorySize(const rocblas_side side,
                                    const rocsolver_eigenvectors howmny,
                                    const I n,
                                    const I batch_count,
                                    size_t* size_X,
                                    size_t* size_tmp,
                                    size_t* size_R,
                                    size_t* size_tmax,
                                    size_t* size_workArr)
{
    using S = decltype(std::real(T{}));

    // quick return
    if(n == 0 || batch_count == 0)
    {
        *size_X = 0;
        *size_tmp = 0;
        *size_R = 0;
        *size_tmax = 0;
        *size_workArr = 0;
        return;
    }

    const I nc = std::min(n, I(TREVC3_NC));
    const bool over = (howmny == rocsolver_eigenvectors_backtransform);

    // vectors of a block of eigenvectors, their back-transformation, and the
    // right-hand sides of the diagonal-block solves
    *size_X = sizeof(T) * n * nc * batch_count;
    *size_tmp = over ? sizeof(T) * n * nc * batch_count : 0;
    *size_R = sizeof(T) * TREVC3_NB * nc * batch_count;
    *size_tmax = sizeof(S) * batch_count;

    // pointer arrays for the matrix-matrix products of the batched version
    *size_workArr = BATCHED ? sizeof(T*) * 2 * batch_count : 0;
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename U>
rocblas_status rocsolver_trevc3_template(rocblas_handle handle,
                                         const rocblas_side side,
                                         const rocsolver_eigenvectors howmny,
                                         const I n,
                                         U A,
                                         const rocblas_stride shiftT,
                                         const I ldt,
                                         const rocblas_stride strideT,
                                         U VL,
                                         const rocblas_stride shiftVL,
                                         const I ldvl,
                                         const rocblas_stride strideVL,
                                         U VR,
                                         const rocblas_stride shiftVR,
                                         const I ldvr,
                                         const rocblas_stride strideVR,
                                         const I batch_count,
                                         T* X,
                                         T* tmp,
                                         T* R,
                                         void* tmax,
                                         T** workArr)
{
    ROCSOLVER_ENTER("trevc3", "side:", side, "howmny:", howmny, "n:", n, "shiftT:", shiftT,
                    "ldt:", ldt, "shiftVL:", shiftVL, "ldvl:", ldvl, "shiftVR:", shiftVR,
                    "ldvr:", ldvr, "bc:", batch_count);

    using S = decltype(std::real(T{}));

    // quick return
    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    rocblas_pointer_mode_saver saver(handle, rocblas_pointer_mode_host);

    constexpr I NB = TREVC3_NB;
    constexpr int BS = TREVC3_BLOCKSIZE;
    const bool leftv = (side != rocblas_side_right);
    const bool rightv = (side != rocblas_side_left);
    const bool over = (howmny == rocsolver_eigenvectors_backtransform);
    const I ncmax = std::min(n, I(TREVC3_NC));
    const rocblas_stride strideX = rocblas_stride(n) * ncmax;
    const rocblas_stride strideR = rocblas_stride(NB) * ncmax;
    S* tmaxS = reinterpret_cast<S*>(tmax);

    const T one = T(1);
    const T mone = T(-1);
    const T zero = T(0);

    // bound on the row and column sums of the strictly upper triangular part of T
    ROCBLAS_CHECK(trevc3_norms<T>(handle, stream, n, A, shiftT, ldt, strideT, batch_count, tmaxS));

    if(rightv)
    {
        // blocks of eigenvectors from the last one (the back-transformation of block
        // p0:p1-1 only reads the columns 0:p1-1 of VR)
        const I nblk = (n - 1) / ncmax + 1;
        for(I kb = nblk - 1; kb >= 0; kb--)
        {
            const I p0 = kb * ncmax;
            const I p1 = std::min(p0 + ncmax, n);
            const I nc = p1 - p0;
            HIP_CHECK(hipMemsetAsync(X, 0, sizeof(T) * strideX * batch_count, stream));

            // diagonal blocks from the bottom
            for(I i0 = ((p1 - 1) / NB) * NB; i0 >= 0; i0 -= NB)
            {
                const I i1 = std::min(i0 + NB, p1);
                const I nb = i1 - i0;

                // coupling with the rows below: R = -T(i0:i1-1, i1:p1-1) * X(i1:p1-1, :)
                // for the eigenvectors k >= i1
                const I kc0 = std::max(i1, p0) - p0;
                if(i1 < p1)
                    rocblasCall_gemm(handle, rocblas_operation_none, rocblas_operation_none, nb,
                                     nc - kc0, p1 - i1, &mone, A, shiftT + idx2D(i0, i1, ldt), ldt,
                                     strideT, X, idx2D(i1, kc0, n), n, strideX, &zero, R,
                                     idx2D(0, kc0, NB), NB, strideR, batch_count, workArr);

                const I ncols = p1 - std::max(i0, p0);
                ROCSOLVER_LAUNCH_KERNEL((trevc3_solve_kernel<false, NB, BS, T>),
                                        dim3((ncols - 1) / BS + 1, batch_count), dim3(BS), 0,
                                        stream, n, A, shiftT, ldt, strideT, i0, nb, p0, p1, R,
                                        strideR, X, strideX, tmaxS);
            }

            // back-transformation and normalization
            T* src = X;
            if(over)
            {
                ROCSOLVER_LAUNCH_KERNEL((trevc3_scale_kernel<BS, T>), dim3(nc, batch_count),
                                        dim3(BS), 0, stream, false, n, p0, X, strideX);
                rocblasCall_gemm(handle, rocblas_operation_none, rocblas_operation_none, n, nc, p1,
                                 &one, VR, shiftVR, ldvr, strideVR, X, 0, n, strideX, &zero, tmp, 0,
                                 n, strideX, batch_count, workArr);
                src = tmp;
            }
            ROCSOLVER_LAUNCH_KERNEL((trevc3_normalize_kernel<BS, T>), dim3(nc, batch_count),
                                    dim3(BS), 0, stream, !over, false, n, p0, src, strideX, VR,
                                    shiftVR, ldvr, strideVR);
        }
    }

    if(leftv)
    {
        // blocks of eigenvectors from the first one (the back-transformation of block
        // p0:p1-1 only reads the columns p0:n-1 of VL)
        for(I p0 = 0; p0 < n; p0 += ncmax)
        {
            const I p1 = std::min(p0 + ncmax, n);
            const I nc = p1 - p0;
            HIP_CHECK(hipMemsetAsync(X, 0, sizeof(T) * strideX * batch_count, stream));

            // diagonal blocks from the top
            for(I i0 = (p0 / NB) * NB; i0 < n; i0 += NB)
            {
                const I i1 = std::min(i0 + NB, n);
                const I nb = i1 - i0;

                // coupling with the rows above: R = -T(p0:i0-1, i0:i1-1)^H * X(p0:i0-1, :)
                // for the eigenvectors k < i0
                const I ncr = std::min(i0, p1) - p0;
                if(ncr > 0)
                    rocblasCall_gemm(handle, rocblas_operation_conjugate_transpose,
                                     rocblas_operation_none, nb, ncr, i0 - p0, &mone, A,
                                     shiftT + idx2D(p0, i0, ldt), ldt, strideT, X, idx2D(p0, 0, n),
                                     n, strideX, &zero, R, 0, NB, strideR, batch_count, workArr);

                const I ncols = std::min(i1, p1) - p0;
                ROCSOLVER_LAUNCH_KERNEL((trevc3_solve_kernel<true, NB, BS, T>),
                                        dim3((ncols - 1) / BS + 1, batch_count), dim3(BS), 0,
                                        stream, n, A, shiftT, ldt, strideT, i0, nb, p0, p1, R,
                                        strideR, X, strideX, tmaxS);
            }

            // back-transformation and normalization
            T* src = X;
            if(over)
            {
                ROCSOLVER_LAUNCH_KERNEL((trevc3_scale_kernel<BS, T>), dim3(nc, batch_count),
                                        dim3(BS), 0, stream, true, n, p0, X, strideX);
                rocblasCall_gemm(handle, rocblas_operation_none, rocblas_operation_none, n, nc,
                                 n - p0, &one, VL, shiftVL + idx2D(0, p0, ldvl), ldvl, strideVL, X,
                                 idx2D(p0, 0, n), n, strideX, &zero, tmp, 0, n, strideX,
                                 batch_count, workArr);
                src = tmp;
            }
            ROCSOLVER_LAUNCH_KERNEL((trevc3_normalize_kernel<BS, T>), dim3(nc, batch_count),
                                    dim3(BS), 0, stream, false, !over, n, p0, src, strideX, VL,
                                    shiftVL, ldvl, strideVL);
        }
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE

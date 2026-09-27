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

/*
 * ===========================================================================
 *    GEBAL follows the reference implementation of LAPACK 3.12 step by step,
 *    so that ilo, ihi, scale and the balanced matrix match those of the host
 *    LAPACK, except for rounding in the row and column norms (and in the presence
 *    of NaN, see below). Each matrix of the batch is processed by a single thread-block;
 *    the loops over rows and columns are sequential (as in LAPACK) and the
 *    work on a single row or column is spread over the threads of the block.
 * ===========================================================================
 */

/** GEBAL_SCALED_SQUARE returns |a * 2^(-e)|^2. Scaling by a power of two is exact
    (unless the result underflows), so it prevents overflow without changing
    the relative size of the entries. **/
template <typename S, typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
__device__ inline S gebal_scaled_square(const T a, const int e)
{
    S x = ldexp(a, -e);
    return x * x;
}

template <typename S, typename T, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
__device__ inline S gebal_scaled_square(const T a, const int e)
{
    S x = ldexp(a.real(), -e);
    S y = ldexp(a.imag(), -e);
    return x * x + y * y;
}

/** GEBAL_SQABS returns |a|^2. **/
template <typename S, typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
__device__ inline S gebal_sqabs(const T a)
{
    return a * a;
}

template <typename S, typename T, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
__device__ inline S gebal_sqabs(const T a)
{
    return a.real() * a.real() + a.imag() * a.imag();
}

/** GEBAL_VEC_INFO accumulates, in a single pass over a row or column, the data
    needed by the balancing step:
    - the position of the first entry of largest |Re|+|Im| (as returned by I_AMAX)
      and its absolute value,
    - the sum of squares of the entries in the active range. As long as these entries
      lie in [safe_lo, safe_hi], the plain sum cannot overflow or lose accuracy by
      underflow; otherwise unsafe is set, and the norm is recomputed with
      gebal_scaled_ssq. **/
template <typename S, typename I>
struct gebal_vec_info
{
    S amax;
    I iamax;
    S mod;
    S ssq;
    int unsafe;

    static __device__ constexpr S safe_lo()
    {
        return std::is_same<S, float>::value ? S(0x1p-50f) : S(0x1p-500);
    }
    static __device__ constexpr S safe_hi()
    {
        return std::is_same<S, float>::value ? S(0x1p50f) : S(0x1p500);
    }

    __device__ void init(const I n)
    {
        amax = S(-1);
        iamax = n;
        mod = 0;
        ssq = 0;
        unsafe = 0;
    }

    template <typename T>
    __device__ void add(const T a, const I pos, const bool in_range)
    {
        S v = aabs<S>(a);
        // (a NaN is never the maximum here; LAPACK IZAMAX returns it when it is the first
        // entry of its range, which then makes the balancing stop with info = -3)
        if(v > amax)
        {
            amax = v;
            iamax = pos;
            mod = std::abs(a);
        }
        // (a NaN in the norm range makes the norm NaN, as in LAPACK, so that the
        // balancing stops)
        if(in_range)
        {
            if(v > safe_hi() || (v < safe_lo() && v > 0))
                unsafe = 1;
            else
                ssq += gebal_sqabs<S>(a);
        }
    }

    __device__ void combine(const S amax2, const I iamax2, const S mod2, const S ssq2, const int unsafe2)
    {
        if(amax2 > amax || (amax2 == amax && iamax2 < iamax))
        {
            amax = amax2;
            iamax = iamax2;
            mod = mod2;
        }
        ssq += ssq2;
        unsafe |= unsafe2;
    }

    // butterfly reduction across the wavefront (valid for wave32 and wave64);
    // as the combination is commutative, all the lanes end up with the same values
    __device__ void wave_reduce()
    {
        for(int offset = warpSize / 2; offset > 0; offset /= 2)
            combine(__shfl_xor(amax, offset), __shfl_xor(iamax, offset), __shfl_xor(mod, offset),
                    __shfl_xor(ssq, offset), __shfl_xor(unsafe, offset));
    }
};

/** GEBAL_SCALED_SSQ computes the 2-norm of a vector as ssq * 2^(2e), where the exponent e
    follows the largest entry seen so far. It is used only when some entries are too
    large or too small for a plain sum of squares. **/
template <typename S>
struct gebal_scaled_ssq
{
    int e;
    S ssq;

    __device__ void init()
    {
        e = -4096;
        ssq = 0;
    }

    template <typename T>
    __device__ void add(const T a)
    {
        S v = aabs<S>(a);
        if(v > 0)
        {
            int ev;
            frexp(v, &ev);
            if(ev > e)
            {
                ssq = ldexp(ssq, 2 * (e - ev));
                e = ev;
            }
            ssq += gebal_scaled_square<S>(a, e);
        }
        else if(std::isnan(v))
            ssq = v;
    }

    __device__ void combine(const int e2, const S ssq2)
    {
        if(e2 > e)
        {
            ssq = ldexp(ssq, 2 * (e - e2)) + ssq2;
            e = e2;
        }
        else
            ssq += ldexp(ssq2, 2 * (e2 - e));
    }

    __device__ void wave_reduce()
    {
        for(int offset = warpSize / 2; offset > 0; offset /= 2)
            combine(__shfl_xor(e, offset), __shfl_xor(ssq, offset));
    }

    __device__ S norm() const
    {
        return ldexp(sqrt(ssq), e);
    }
};

/** GEBAL_SWAP interchanges rows or columns p and q of the active part of A.
    The column swap (over rows 0:l) and the row swap (over columns k:n-1) share
    the entries (p,p), (p,q), (q,p) and (q,q), so they must be separated by
    a barrier. **/
template <int BS, typename T, typename I>
__device__ void
    gebal_swap(const I tid, const I n, const I k, const I l, const I p, const I q, T* A, const I lda)
{
    for(I r = tid; r <= l; r += BS)
        swap(A[idx2D(r, p, lda)], A[idx2D(r, q, lda)]);
    __syncthreads();

    for(I c = k + tid; c < n; c += BS)
        swap(A[idx2D(p, c, lda)], A[idx2D(q, c, lda)]);
    __syncthreads();
}

/** GEBAL_BLOCK_MAX returns the maximum of v over the thread-block to all the threads.
    It uses one of two shared buffers alternately (selected by buf), so that
    one barrier per call is enough. **/
template <int BS, typename I>
__device__ I gebal_block_max(I v, I (*s_red)[BS / 32], int& buf)
{
    const int lane = hipThreadIdx_x % warpSize;
    const int wave = hipThreadIdx_x / warpSize;
    const int nwaves = BS / warpSize;

    for(int offset = warpSize / 2; offset > 0; offset /= 2)
        v = std::max(v, I(__shfl_xor(v, offset)));
    if(lane == 0)
        s_red[buf][wave] = v;
    __syncthreads();
    v = s_red[buf][0];
    for(int w = 1; w < nwaves; w++)
        v = std::max(v, s_red[buf][w]);
    buf = 1 - buf;
    return v;
}

/** GEBAL_KERNEL balances each matrix in the batch.
    Launch with one thread-block of BS threads per matrix. BS must be a multiple of 64. **/
template <int BS, typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(BS) gebal_kernel(const rocsolver_balance job,
                                                         const I n,
                                                         U AA,
                                                         const rocblas_stride shiftA,
                                                         const I lda,
                                                         const rocblas_stride strideA,
                                                         I* iloA,
                                                         I* ihiA,
                                                         S* scaleA,
                                                         const rocblas_stride strideS,
                                                         I* workA)
{
    const I bid = hipBlockIdx_x;
    const I tid = hipThreadIdx_x;

    // quick return
    if(n == 0)
    {
        if(tid == 0)
        {
            iloA[bid] = 1;
            ihiA[bid] = 0;
        }
        return;
    }

    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    S* scale = scaleA + rocblas_stride(bid) * strideS;

    if(job == rocsolver_balance_none)
    {
        for(I i = tid; i < n; i += BS)
            scale[i] = S(1);
        if(tid == 0)
        {
            iloA[bid] = 1;
            ihiA[bid] = n;
        }
        return;
    }

    // Permutation to isolate eigenvalues if possible.
    // The active submatrix is A(k:l, k:l) (0-based).
    I k = 0;
    I l = n - 1;

    if(job != rocsolver_balance_scale)
    {
        // To avoid reading a whole row or column each time that LAPACK tests whether
        // it isolates an eigenvalue, cnt keeps the number of non-zero off-diagonal entries
        // of each row (or column) in the active range. The counts are updated when the
        // active range shrinks; swaps simply interchange two counts. The rows and columns
        // are visited in the same order as in LAPACK, so the permutation is the same.
        I* cnt = workA + rocblas_stride(bid) * n;
        constexpr int MAX_WAVES = BS / 32;
        __shared__ I s_found[2][MAX_WAVES];
        int buf = 0;

        // cnt[r] = number of non-zeros in A(r, 0:l), excluding the diagonal
        // (one thread per row, so that the reads are coalesced)
        for(I r = tid; r <= l; r += BS)
        {
            I c = 0;
            for(I j = 0; j <= l; j++)
                c += (j != r && A[idx2D(r, j, lda)] != T(0));
            cnt[r] = c;
        }
        __syncthreads();

        // Search for rows isolating an eigenvalue and push them down.
        bool noconv = true;
        while(noconv)
        {
            noconv = false;
            // (as in LAPACK, the scan starts at the value of l on entry and moves up,
            // while l decreases)
            I i = l;
            while(i >= 0)
            {
                // find the next row i' <= i with cnt[i'] == 0
                I found = -1;
                for(I base = i; base >= 0 && found < 0; base -= BS)
                {
                    I r = base - tid;
                    found = gebal_block_max<BS>((r >= 0 && cnt[r] == 0) ? r : I(-1), s_found, buf);
                }
                if(found < 0)
                    break;
                i = found;

                if(tid == 0)
                {
                    scale[l] = S(i + 1);
                    swap(cnt[i], cnt[l]);
                }
                if(i != l)
                    gebal_swap<BS>(tid, n, k, l, i, l, A, lda);
                noconv = true;

                if(l == 0)
                {
                    if(tid == 0)
                    {
                        iloA[bid] = 1;
                        ihiA[bid] = 1;
                    }
                    return;
                }

                // column l leaves the active range
                __syncthreads();
                for(I r = tid; r < l; r += BS)
                    if(A[idx2D(r, l, lda)] != T(0))
                        cnt[r]--;
                __syncthreads();

                l--;
                i--;
            }
        }

        // cnt[j] = number of non-zeros in A(k:l, j), excluding the diagonal
        // (one wavefront per column, so that the reads are coalesced)
        {
            const int lane = tid % warpSize;
            const int wave = tid / warpSize;
            const int nwaves = BS / warpSize;
            for(I j = k + wave; j <= l; j += nwaves)
            {
                I c = 0;
                for(I r = k + lane; r <= l; r += warpSize)
                    c += (r != j && A[idx2D(r, j, lda)] != T(0));
                for(int offset = warpSize / 2; offset > 0; offset /= 2)
                    c += __shfl_xor(c, offset);
                if(lane == 0)
                    cnt[j] = c;
            }
        }
        __syncthreads();

        // Search for columns isolating an eigenvalue and push them left.
        noconv = true;
        while(noconv)
        {
            noconv = false;
            // (as in LAPACK, the scan starts at the value of k on entry and moves down,
            // while k increases)
            I j = k;
            while(j <= l)
            {
                // find the next column j' >= j with cnt[j'] == 0
                // (the minimum is found as the maximum of -j')
                I found = -1;
                for(I base = j; base <= l && found < 0; base += BS)
                {
                    I c = base + tid;
                    I m = gebal_block_max<BS>((c <= l && cnt[c] == 0) ? -c : -(n + 1), s_found, buf);
                    found = (m == -(n + 1)) ? -1 : -m;
                }
                if(found < 0)
                    break;
                j = found;

                if(tid == 0)
                {
                    scale[k] = S(j + 1);
                    swap(cnt[j], cnt[k]);
                }
                if(j != k)
                    gebal_swap<BS>(tid, n, k, l, j, k, A, lda);
                noconv = true;

                // row k leaves the active range
                __syncthreads();
                for(I c = k + 1 + tid; c <= l; c += BS)
                    if(A[idx2D(k, c, lda)] != T(0))
                        cnt[c]--;
                __syncthreads();

                k++;
                j++;
            }
        }
    }

    // Initialize scale for the non-permuted submatrix.
    for(I i = k + tid; i <= l; i += BS)
        scale[i] = S(1);

    if(job == rocsolver_balance_permute)
    {
        if(tid == 0)
        {
            iloA[bid] = k + 1;
            ihiA[bid] = l + 1;
        }
        return;
    }

    // Balance the submatrix in rows k to l.
    // Iterative loop for norm reduction.
    const S sclfac = S(2);
    const S factor = S(0.95);
    const S sfmin1 = std::numeric_limits<S>::min() / std::numeric_limits<S>::epsilon();
    const S sfmax1 = S(1) / sfmin1;
    const S sfmin2 = sfmin1 * sclfac;
    const S sfmax2 = S(1) / sfmin2;

    // workspace for the reductions across wavefronts. Two buffers are used alternately,
    // so that a single barrier per reduction is enough.
    constexpr int MAX_WAVES = BS / 32;
    __shared__ S s_amax[2][2][MAX_WAVES], s_mod[2][2][MAX_WAVES], s_ssq[2][2][MAX_WAVES];
    __shared__ I s_iamax[2][2][MAX_WAVES];
    __shared__ int s_unsafe[2][2][MAX_WAVES];
    __shared__ S s_sssq[2][2][MAX_WAVES];
    __shared__ int s_se[2][2][MAX_WAVES];

    const int lane = tid % warpSize;
    const int wave = tid / warpSize;
    const int nwaves = BS / warpSize;
    int buf = 0;
    int sbuf = 0;

    // scale is written by several threads above
    __syncthreads();

    bool noconv = true;
    bool nan_found = false;
    while(noconv && !nan_found)
    {
        noconv = false;

        for(I i = k; i <= l; i++)
        {
            // Single pass over column i and row i to get:
            // c = ||A(k:l, i)||, ca = |A(ica, i)| with ica = I_AMAX(A(0:l, i)),
            // r = ||A(i, k:l)||, ra = |A(i, ira)| with ira = I_AMAX(A(i, k:n-1)).
            gebal_vec_info<S, I> col, row;
            col.init(n);
            row.init(n);
            const S si = scale[i];

#pragma unroll 4
            for(I rr = tid; rr <= l; rr += BS)
                col.add(A[idx2D(rr, i, lda)], rr, rr >= k);
#pragma unroll 4
            for(I cc = k + tid; cc < n; cc += BS)
                row.add(A[idx2D(i, cc, lda)], cc, cc <= l);

            col.wave_reduce();
            row.wave_reduce();
            if(lane == 0)
            {
                s_amax[buf][0][wave] = col.amax;
                s_iamax[buf][0][wave] = col.iamax;
                s_mod[buf][0][wave] = col.mod;
                s_ssq[buf][0][wave] = col.ssq;
                s_unsafe[buf][0][wave] = col.unsafe;
                s_amax[buf][1][wave] = row.amax;
                s_iamax[buf][1][wave] = row.iamax;
                s_mod[buf][1][wave] = row.mod;
                s_ssq[buf][1][wave] = row.ssq;
                s_unsafe[buf][1][wave] = row.unsafe;
            }
            // (after this barrier, A(:,i), A(i,:) and scale[i] are no longer read in this step,
            // except by the fallback below, so they can be updated without a further barrier)
            __syncthreads();
            // every wavefront combines the partial results of all the wavefronts
            col.init(n);
            row.init(n);
            if(lane < nwaves)
            {
                col.combine(s_amax[buf][0][lane], s_iamax[buf][0][lane], s_mod[buf][0][lane],
                            s_ssq[buf][0][lane], s_unsafe[buf][0][lane]);
                row.combine(s_amax[buf][1][lane], s_iamax[buf][1][lane], s_mod[buf][1][lane],
                            s_ssq[buf][1][lane], s_unsafe[buf][1][lane]);
            }
            col.wave_reduce();
            row.wave_reduce();
            buf = 1 - buf;

            S c = sqrt(col.ssq);
            S r = sqrt(row.ssq);

            // fallback: recompute the norms with scaling if some entries are too large
            // or too small
            if(col.unsafe || row.unsafe)
            {
                gebal_scaled_ssq<S> cs, rs;
                cs.init();
                rs.init();
                if(col.unsafe)
                    for(I rr = k + tid; rr <= l; rr += BS)
                        cs.add(A[idx2D(rr, i, lda)]);
                if(row.unsafe)
                    for(I cc = k + tid; cc <= l; cc += BS)
                        rs.add(A[idx2D(i, cc, lda)]);
                cs.wave_reduce();
                rs.wave_reduce();
                if(lane == 0)
                {
                    s_se[sbuf][0][wave] = cs.e;
                    s_sssq[sbuf][0][wave] = cs.ssq;
                    s_se[sbuf][1][wave] = rs.e;
                    s_sssq[sbuf][1][wave] = rs.ssq;
                }
                __syncthreads();
                cs.init();
                rs.init();
                if(lane < nwaves)
                {
                    cs.combine(s_se[sbuf][0][lane], s_sssq[sbuf][0][lane]);
                    rs.combine(s_se[sbuf][1][lane], s_sssq[sbuf][1][lane]);
                }
                cs.wave_reduce();
                rs.wave_reduce();
                sbuf = 1 - sbuf;

                if(col.unsafe)
                    c = cs.norm();
                if(row.unsafe)
                    r = rs.norm();
            }

            S ca = col.mod;
            S ra = row.mod;

            // Guard against zero c or r due to underflow.
            if(c == 0 || r == 0)
                continue;

            // Exit if NaN to avoid infinite loop.
            // (LAPACK returns with info = -3 and does not set ilo and ihi; here the
            // balancing stops, and ilo and ihi are set.)
            if(std::isnan(c + ca + r + ra))
            {
                nan_found = true;
                break;
            }

            S g = r / sclfac;
            S f = S(1);
            S s = c + r;

            while(c < g && std::max(f, std::max(c, ca)) < sfmax2
                  && std::min(r, std::min(g, ra)) > sfmin2)
            {
                f = f * sclfac;
                c = c * sclfac;
                ca = ca * sclfac;
                r = r / sclfac;
                g = g / sclfac;
                ra = ra / sclfac;
            }

            g = c / sclfac;

            while(g >= r && std::max(r, ra) < sfmax2
                  && std::min(std::min(f, c), std::min(g, ca)) > sfmin2)
            {
                f = f / sclfac;
                c = c / sclfac;
                g = g / sclfac;
                ca = ca / sclfac;
                r = r * sclfac;
                ra = ra * sclfac;
            }

            // Now balance.
            if((c + r) >= factor * s)
                continue;
            if(f < S(1) && si < S(1))
            {
                if(f * si <= sfmin1)
                    continue;
            }
            if(f > S(1) && si > S(1))
            {
                if(si >= sfmax1 / f)
                    continue;
            }
            g = S(1) / f;
            noconv = true;

            // Scale row i by g and column i by f. A(i,i) is scaled by both, in the same
            // order as LAPACK, by a single thread.
            for(I cc = k + tid; cc < n; cc += BS)
                if(cc != i)
                    A[idx2D(i, cc, lda)] *= g;
            for(I rr = tid; rr <= l; rr += BS)
                if(rr != i)
                    A[idx2D(rr, i, lda)] *= f;
            if(tid == 0)
            {
                T aii = A[idx2D(i, i, lda)];
                aii *= g;
                aii *= f;
                A[idx2D(i, i, lda)] = aii;
                scale[i] = si * f;
            }
            __syncthreads();
        }
    }

    if(tid == 0)
    {
        iloA[bid] = k + 1;
        ihiA[bid] = l + 1;
    }
}

template <typename T, typename I>
void rocsolver_gebal_getMemorySize(const rocsolver_balance job,
                                   const I n,
                                   const I batch_count,
                                   size_t* size_work)
{
    // counts of non-zeros used by the permutation step
    if(n == 0 || batch_count == 0 || job == rocsolver_balance_none || job == rocsolver_balance_scale)
        *size_work = 0;
    else
        *size_work = sizeof(I) * n * batch_count;
}

template <typename T, typename I, typename S>
rocblas_status rocsolver_gebal_argCheck(rocblas_handle handle,
                                        const rocsolver_balance job,
                                        const I n,
                                        const I lda,
                                        T A,
                                        I* ilo,
                                        I* ihi,
                                        S* scale,
                                        const I batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(job != rocsolver_balance_none && job != rocsolver_balance_permute
       && job != rocsolver_balance_scale && job != rocsolver_balance_both)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < n || lda < 1 || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !scale) || (batch_count && !ilo) || (batch_count && !ihi))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <bool BATCHED, bool STRIDED, typename T, typename I, typename S, typename U>
rocblas_status rocsolver_gebal_template(rocblas_handle handle,
                                        const rocsolver_balance job,
                                        const I n,
                                        U A,
                                        const rocblas_stride shiftA,
                                        const I lda,
                                        const rocblas_stride strideA,
                                        I* ilo,
                                        I* ihi,
                                        S* scale,
                                        const rocblas_stride strideS,
                                        const I batch_count,
                                        I* work)
{
    ROCSOLVER_ENTER("gebal", "job:", job, "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // (when n == 0 the kernel only sets ilo = 1 and ihi = 0)
    ROCSOLVER_LAUNCH_KERNEL((gebal_kernel<GEBAL_BLOCKSIZE, T>), dim3(batch_count),
                            dim3(GEBAL_BLOCKSIZE), 0, stream, job, n, A, shiftA, lda, strideA, ilo,
                            ihi, scale, strideS, work);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE

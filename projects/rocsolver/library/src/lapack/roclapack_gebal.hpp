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

/** GEBAL_MOD returns |a|. As LAPACK (ABS), it returns Inf when a part of a is infinite, even
    if the other part is infinite or NaN. **/
template <typename S, typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
__device__ inline S gebal_mod(const T a)
{
    return std::abs(a);
}

template <typename S, typename T, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
__device__ inline S gebal_mod(const T a)
{
    if(std::isinf(a.real()) || std::isinf(a.imag()))
        return std::numeric_limits<S>::infinity();
    return std::abs(a);
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
            mod = gebal_mod<S>(a);
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

    // starts from the plain sum of squares s of entries in [safe_lo, safe_hi] (see
    // gebal_vec_info), which cannot overflow or lose accuracy by underflow
    __device__ void init_plain(const S s)
    {
        init();
        if(s != 0)
        {
            e = 0;
            ssq = s;
        }
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

/** GEBAL_FACTOR returns the power of 2 f by which LAPACK scales column i (and row i by 1/f),
    from the 2-norms c and r of the column and row in the active range, the largest moduli
    ca and ra of their entries, and the current factor si of i: f = 1 if i is not scaled,
    and f = 0 if one of the values is NaN (then the balancing stops). **/
template <typename S>
__device__ S gebal_factor(S c, S r, S ca, S ra, const S si)
{
    const S sclfac = S(2);
    const S factor = S(0.95);
    const S sfmin1 = std::numeric_limits<S>::min() / std::numeric_limits<S>::epsilon();
    const S sfmax1 = S(1) / sfmin1;
    const S sfmin2 = sfmin1 * sclfac;
    const S sfmax2 = S(1) / sfmin2;

    // Guard against zero c or r due to underflow.
    if(c == 0 || r == 0)
        return S(1);

    // Exit if NaN to avoid infinite loop.
    // (LAPACK returns with info = -3 and does not set ilo and ihi; here the
    // balancing stops, and ilo and ihi are set.)
    if(std::isnan(c + ca + r + ra))
        return S(0);

    S g = r / sclfac;
    S f = S(1);
    S s = c + r;

    while(c < g && std::max(f, std::max(c, ca)) < sfmax2 && std::min(r, std::min(g, ra)) > sfmin2)
    {
        f = f * sclfac;
        c = c * sclfac;
        ca = ca * sclfac;
        r = r / sclfac;
        g = g / sclfac;
        ra = ra / sclfac;
    }

    g = c / sclfac;

    while(g >= r && std::max(r, ra) < sfmax2 && std::min(std::min(f, c), std::min(g, ca)) > sfmin2)
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
        return S(1);
    if(f < S(1) && si < S(1))
    {
        if(f * si <= sfmin1)
            return S(1);
    }
    if(f > S(1) && si > S(1))
    {
        if(si >= sfmax1 / f)
            return S(1);
    }
    return f;
}

/** GEBAL_ROW_SEARCH searches for the rows that isolate an eigenvalue and moves them to the
    bottom of the active submatrix A(k:l, k:l) (0-based), in the order of LAPACK; cnt holds
    the number of non-zero off-diagonal entries of each row in the active range (it is
    updated). It returns true if all the eigenvalues are isolated (then l = 0). All the
    threads of the thread-block call it. **/
template <int BS, typename T, typename I, typename S>
__device__ bool gebal_row_search(const I n,
                                 T* A,
                                 const I lda,
                                 I* cnt,
                                 const I k,
                                 I& l,
                                 S* scale,
                                 I (*s_found)[BS / 32],
                                 int& buf)
{
    const I tid = hipThreadIdx_x;
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
                return true;

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

    return false;
}

/** GEBAL_COL_SEARCH searches for the columns that isolate an eigenvalue and moves them to
    the left of the active submatrix A(k:l, k:l) (0-based), in the order of LAPACK; cnt
    holds the number of non-zero off-diagonal entries of each column in the active range
    (it is updated). All the threads of the thread-block call it. **/
template <int BS, typename T, typename I, typename S>
__device__ void gebal_col_search(const I n,
                                 T* A,
                                 const I lda,
                                 I* cnt,
                                 I& k,
                                 const I l,
                                 S* scale,
                                 I (*s_found)[BS / 32],
                                 int& buf)
{
    const I tid = hipThreadIdx_x;
    bool noconv = true;
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
        if(gebal_row_search<BS>(n, A, lda, cnt, k, l, scale, s_found, buf))
        {
            if(tid == 0)
            {
                iloA[bid] = 1;
                ihiA[bid] = 1;
            }
            return;
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
        gebal_col_search<BS>(n, A, lda, cnt, k, l, scale, s_found, buf);
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

            // (f = 0: NaN, the balancing stops; f = 1: no scaling)
            const S f = gebal_factor(c, r, col.mod, row.mod, si);
            if(f == S(0))
            {
                nan_found = true;
                break;
            }
            if(f == S(1))
                continue;
            const S g = S(1) / f;
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

/*
 * ===========================================================================
 *    Multi-block path (n >= GEBAL_MULTI_MIN). The steps are those of gebal_kernel, in
 *    the same order, so that the results are the same:
 *    - permutation: the counts of non-zeros of the rows and columns are computed by many
 *      thread-blocks (gebal_mb_rowcnt_kernel, gebal_mb_colcnt_kernel), and the searches
 *      run on one thread-block per matrix (gebal_row_search, gebal_col_search);
 *    - scaling: the rows and columns i = k:l are visited in batches of GEBAL_BATCH. For
 *      each batch, gebal_mb_stats_kernel computes the norms and maxima of the parts of
 *      its columns and rows outside the diagonal block of the batch (with many
 *      thread-blocks, in GEBAL_NSEG segments); these parts do not change while the batch
 *      is processed. gebal_mb_decide_kernel then takes the decisions in order, as in
 *      LAPACK, on one wavefront per matrix, adding the entries of the diagonal block
 *      (kept in shared memory and scaled as the decisions are taken), and
 *      gebal_mb_apply_kernel scales the parts outside the block. Each sweep is
 *      followed by a synchronization with the host, which starts another sweep while a
 *      matrix has been scaled.
 * ===========================================================================
 */

/** GEBAL_PART holds the partial results of gebal_mb_stats_kernel for one column or row
    of the batch and one segment (see gebal_vec_info and gebal_scaled_ssq; when no entry
    is too large or too small, the scaled sum is the plain sum with exponent 0). **/
template <typename S, typename I>
struct gebal_part
{
    S amax;
    S mod;
    S ssq;
    S sssq;
    I iamax;
    int unsafe;
    int se;
};

/** GEBAL_MB_LAYOUT is the layout of the workspace of the multi-block path (in bytes):
    per matrix, k, l, stop (1: NaN found, 2: all the eigenvalues isolated, 3: converged),
    noconv and any (the batch scaled a row or column); the factors of the batch; the counts
    of the permutation step; the partial results of the batch. **/
template <typename S, typename I>
struct gebal_mb_layout
{
    size_t kk, ll, stop, noconv, any, fac, cnt, part, size;

    static __host__ __device__ size_t al(const size_t x)
    {
        return ((x + 255) / 256) * 256;
    }

    __host__ __device__ gebal_mb_layout(const I n, const I bc)
    {
        kk = 0;
        ll = kk + al(sizeof(I) * bc);
        stop = ll + al(sizeof(I) * bc);
        noconv = stop + al(sizeof(I) * bc);
        any = noconv + al(sizeof(I) * bc);
        fac = any + al(sizeof(I) * bc);
        cnt = fac + al(sizeof(S) * GEBAL_BATCH * bc);
        part = cnt + al(sizeof(I) * n * bc);
        size = part + al(sizeof(gebal_part<S, I>) * 2 * GEBAL_BATCH * GEBAL_NSEG * bc);
    }
};

/** GEBAL_MB_STATE gives the per-matrix arrays of the workspace. **/
template <typename S, typename I>
struct gebal_mb_state
{
    I *k, *l, *stop, *noconv, *any;
    S* fac;
    I* cnt;
    gebal_part<S, I>* part;

    __host__ __device__ gebal_mb_state(char* ws, const I n, const I bc)
    {
        const gebal_mb_layout<S, I> lay(n, bc);
        k = reinterpret_cast<I*>(ws + lay.kk);
        l = reinterpret_cast<I*>(ws + lay.ll);
        stop = reinterpret_cast<I*>(ws + lay.stop);
        noconv = reinterpret_cast<I*>(ws + lay.noconv);
        any = reinterpret_cast<I*>(ws + lay.any);
        fac = reinterpret_cast<S*>(ws + lay.fac);
        cnt = reinterpret_cast<I*>(ws + lay.cnt);
        part = reinterpret_cast<gebal_part<S, I>*>(ws + lay.part);
    }
};

template <typename S, typename I>
ROCSOLVER_KERNEL void gebal_mb_init_kernel(const I n, const I bc, char* ws)
{
    gebal_mb_state<S, I> st(ws, n, bc);
    const I b = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(b < bc)
    {
        st.k[b] = 0;
        st.l[b] = n - 1;
        st.stop[b] = 0;
        st.noconv[b] = 0;
        st.any[b] = 0;
    }
}

/** GEBAL_MB_ROWCNT_KERNEL: cnt[r] = number of non-zeros in A(r, 0:n-1), excluding the
    diagonal (64 rows per thread-block; the threads of a wavefront read contiguous rows). **/
template <int BS, typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(BS) gebal_mb_rowcnt_kernel(const I n,
                                                                   U AA,
                                                                   const rocblas_stride shiftA,
                                                                   const I lda,
                                                                   const rocblas_stride strideA,
                                                                   const I bc,
                                                                   char* ws)
{
    constexpr int RB = 64;
    static_assert(BS % RB == 0, "the block size must be a multiple of 64");
    constexpr int NC = BS / RB;
    gebal_mb_state<S, I> st(ws, n, bc);
    const I bid = hipBlockIdx_y;
    const T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    I* cnt = st.cnt + rocblas_stride(bid) * n;
    const int r = hipThreadIdx_x % RB;
    const int cl = hipThreadIdx_x / RB;
    const I i = I(hipBlockIdx_x) * RB + r;
    I c = 0;
    if(i < n)
        for(I j = cl; j < n; j += NC)
            c += (j != i && A[idx2D(i, j, lda)] != T(0));
    __shared__ I part[BS];
    part[hipThreadIdx_x] = c;
    __syncthreads();
    if(cl == 0 && i < n)
    {
        for(int q = 1; q < NC; q++)
            c += part[r + q * RB];
        cnt[i] = c;
    }
}

/** GEBAL_MB_COLCNT_KERNEL: cnt[j] = number of non-zeros in A(k:l, j), excluding the
    diagonal, for j = k:l (a wavefront per column). **/
template <int BS, typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(BS) gebal_mb_colcnt_kernel(const I n,
                                                                   U AA,
                                                                   const rocblas_stride shiftA,
                                                                   const I lda,
                                                                   const rocblas_stride strideA,
                                                                   const I bc,
                                                                   char* ws)
{
    gebal_mb_state<S, I> st(ws, n, bc);
    const I bid = hipBlockIdx_y;
    if(st.stop[bid])
        return;
    const I k = st.k[bid];
    const I l = st.l[bid];
    const T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    I* cnt = st.cnt + rocblas_stride(bid) * n;
    const int lane = hipThreadIdx_x % warpSize;
    const int nwaves = BS / warpSize;
    for(I j = k + I(hipBlockIdx_x) * nwaves + hipThreadIdx_x / warpSize; j <= l;
        j += I(hipGridDim_x) * nwaves)
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

/** GEBAL_MB_SEARCH_KERNEL runs the search for rows (rows = true) or columns isolating an
    eigenvalue, on one thread-block per matrix, with the counts computed beforehand. **/
template <int BS, typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(BS) gebal_mb_search_kernel(const bool rows,
                                                                   const I n,
                                                                   U AA,
                                                                   const rocblas_stride shiftA,
                                                                   const I lda,
                                                                   const rocblas_stride strideA,
                                                                   S* scaleA,
                                                                   const rocblas_stride strideS,
                                                                   const I bc,
                                                                   char* ws)
{
    gebal_mb_state<S, I> st(ws, n, bc);
    const I bid = hipBlockIdx_x;
    if(st.stop[bid])
        return;
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    S* scale = scaleA + rocblas_stride(bid) * strideS;
    I* cnt = st.cnt + rocblas_stride(bid) * n;
    __shared__ I s_found[2][BS / 32];
    int buf = 0;
    I k = st.k[bid];
    I l = st.l[bid];
    if(rows)
    {
        const bool all = gebal_row_search<BS>(n, A, lda, cnt, k, l, scale, s_found, buf);
        if(hipThreadIdx_x == 0)
        {
            st.l[bid] = l;
            if(all)
                st.stop[bid] = 2;
        }
    }
    else
    {
        gebal_col_search<BS>(n, A, lda, cnt, k, l, scale, s_found, buf);
        if(hipThreadIdx_x == 0)
            st.k[bid] = k;
    }
}

/** GEBAL_MB_SCALE_INIT_KERNEL sets scale(k:l) = 1 (unless all the eigenvalues were
    isolated). **/
template <typename T, typename I, typename S>
ROCSOLVER_KERNEL void
    gebal_mb_scale_init_kernel(const I n, S* scaleA, const rocblas_stride strideS, const I bc, char* ws)
{
    gebal_mb_state<S, I> st(ws, n, bc);
    const I bid = hipBlockIdx_y;
    if(st.stop[bid] == 2)
        return;
    S* scale = scaleA + rocblas_stride(bid) * strideS;
    const I i = st.k[bid] + I(hipBlockIdx_x) * hipBlockDim_x + hipThreadIdx_x;
    if(i <= st.l[bid])
        scale[i] = S(1);
}

/** GEBAL_MB_STATS_KERNEL computes the partial results of segment p = blockIdx.x for the
    columns (blockIdx.y = 0) or rows (1) of the batch ib0:ib1 = [i0, i0+GEBAL_BATCH-1] ∩
    [k, l], without the entries of the diagonal block of the batch: for column j, the
    maximum over the rows 0:l and the norm over k:l; for row j, the maximum over the
    columns k:n-1 and the norm over k:l (as in gebal_kernel). **/
template <int BS, typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(BS) gebal_mb_stats_kernel(const I n,
                                                                  U AA,
                                                                  const rocblas_stride shiftA,
                                                                  const I lda,
                                                                  const rocblas_stride strideA,
                                                                  const I bc,
                                                                  char* ws,
                                                                  const I i0)
{
    constexpr int BW = GEBAL_BATCH;
    constexpr int NSEG = GEBAL_NSEG;
    gebal_mb_state<S, I> st(ws, n, bc);
    const I bid = hipBlockIdx_z;
    if(st.stop[bid])
        return;
    const I k = st.k[bid];
    const I l = st.l[bid];
    const I ib0 = std::max(i0, k);
    const I ib1 = std::min(i0 + BW - 1, l);
    if(ib0 > ib1)
        return;
    const I nb = ib1 - ib0 + 1;
    const T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    const int part = hipBlockIdx_y;
    const I p = hipBlockIdx_x;
    gebal_part<S, I>* rec = st.part + (rocblas_stride(bid) * 2 + part) * BW * NSEG;
    const int tid = hipThreadIdx_x;

    if(part == 0)
    {
        const int lane = tid % warpSize;
        const int wave = tid / warpSize;
        const int nwaves = BS / warpSize;
        const int64_t len = int64_t(l) + 1;
        const I r0 = I(p * len / NSEG);
        const I r1 = I((p + 1) * len / NSEG);
        for(int jj = wave; jj < nb; jj += nwaves)
        {
            gebal_vec_info<S, I> v;
            v.init(n);
            for(I r = r0 + lane; r < r1; r += warpSize)
            {
                if(r >= ib0 && r <= ib1)
                    continue;
                v.add(A[idx2D(r, ib0 + jj, lda)], r, r >= k);
            }
            v.wave_reduce();

            // (second pass with scaling only if some entries are too large or too small)
            gebal_scaled_ssq<S> sc;
            sc.init_plain(v.ssq);
            if(v.unsafe)
            {
                sc.init();
                for(I r = std::max(r0, k) + lane; r < r1; r += warpSize)
                    if(r < ib0 || r > ib1)
                        sc.add(A[idx2D(r, ib0 + jj, lda)]);
                sc.wave_reduce();
            }
            if(lane == 0)
                rec[jj * NSEG + p] = {v.amax, v.mod, v.ssq, sc.ssq, v.iamax, v.unsafe, sc.e};
        }
    }
    else
    {
        constexpr int NCL = BS / BW;
        static_assert(BS % BW == 0, "the block size must be a multiple of GEBAL_BATCH");
        const int r = tid % BW;
        const int cl = tid / BW;
        const int64_t len = int64_t(n) - k;
        const I c0 = k + I(p * len / NSEG);
        const I c1 = k + I((p + 1) * len / NSEG);
        gebal_vec_info<S, I> v;
        v.init(n);
        if(r < nb)
            for(I c = c0 + cl; c < c1; c += NCL)
            {
                if(c >= ib0 && c <= ib1)
                    continue;
                v.add(A[idx2D(ib0 + r, c, lda)], c, c <= l);
            }

        // (second pass with scaling, by the threads that found entries too large or too small)
        gebal_scaled_ssq<S> sc;
        sc.init_plain(v.ssq);
        if(v.unsafe)
        {
            sc.init();
            for(I c = c0 + cl; c < c1 && c <= l; c += NCL)
                if(c < ib0 || c > ib1)
                    sc.add(A[idx2D(ib0 + r, c, lda)]);
        }
        __shared__ gebal_part<S, I> sp[BS];
        sp[tid] = {v.amax, v.mod, v.ssq, sc.ssq, v.iamax, v.unsafe, sc.e};
        __syncthreads();
        if(cl == 0 && r < nb)
        {
            for(int q = 1; q < NCL; q++)
            {
                const gebal_part<S, I>& o = sp[r + q * BW];
                v.combine(o.amax, o.iamax, o.mod, o.ssq, o.unsafe);
                sc.combine(o.se, o.sssq);
            }
            rec[r * NSEG + p] = {v.amax, v.mod, v.ssq, sc.ssq, v.iamax, v.unsafe, sc.e};
        }
    }
}

/** GEBAL_MB_DECIDE_KERNEL takes the decisions of the batch ib0:ib1 in order (on the first
    wavefront of one thread-block per matrix), as gebal_kernel does: the norms and maxima of
    each column and row are those of its part outside the diagonal block of the batch
    (reduced from the partial results in a fixed order) combined with its entries in the
    block, which is kept in shared memory and scaled as the decisions are taken, and then
    written back. The factors are left in fac for gebal_mb_apply_kernel. **/
template <int BS, typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(BS) gebal_mb_decide_kernel(const I n,
                                                                   U AA,
                                                                   const rocblas_stride shiftA,
                                                                   const I lda,
                                                                   const rocblas_stride strideA,
                                                                   S* scaleA,
                                                                   const rocblas_stride strideS,
                                                                   const I bc,
                                                                   char* ws,
                                                                   const I i0)
{
    constexpr int BW = GEBAL_BATCH;
    constexpr int NSEG = GEBAL_NSEG;
    gebal_mb_state<S, I> st(ws, n, bc);
    const I bid = hipBlockIdx_x;
    const int tid = hipThreadIdx_x;
    S* fac = st.fac + rocblas_stride(bid) * BW;
    const I k = st.k[bid];
    const I l = st.l[bid];
    const I ib0 = std::max(i0, k);
    const I ib1 = std::min(i0 + BW - 1, l);
    if(st.stop[bid] || ib0 > ib1)
    {
        if(tid == 0)
            st.any[bid] = 0;
        return;
    }
    const I nb = ib1 - ib0 + 1;
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    S* scale = scaleA + rocblas_stride(bid) * strideS;

    __shared__ T blk[BW * BW];
    __shared__ gebal_part<S, I> red[2][BW];
    __shared__ int s_any;
    for(I idx = tid; idx < nb * nb; idx += BS)
        blk[(idx % nb) + (idx / nb) * BW] = A[idx2D(ib0 + idx % nb, ib0 + idx / nb, lda)];
    static_assert(BS >= 2 * BW, "the block size must be at least 2 * GEBAL_BATCH");
    if(tid < 2 * nb)
    {
        const int part = tid / nb;
        const int jj = tid % nb;
        const gebal_part<S, I>* rec = st.part + ((rocblas_stride(bid) * 2 + part) * BW + jj) * NSEG;
        gebal_vec_info<S, I> v;
        gebal_scaled_ssq<S> sc;
        v.init(n);
        sc.init();
        for(int p = 0; p < NSEG; p++)
        {
            v.combine(rec[p].amax, rec[p].iamax, rec[p].mod, rec[p].ssq, rec[p].unsafe);
            sc.combine(rec[p].se, rec[p].sssq);
        }
        red[part][jj] = {v.amax, v.mod, v.ssq, sc.ssq, v.iamax, v.unsafe, sc.e};
    }
    if(tid < BW)
        fac[tid] = S(1);
    __syncthreads();

    // The decisions are taken in order by the first wavefront: lane q holds the entries
    // (q, jj) and (jj, q) of the block, so that the contribution of the block to column
    // and row jj is a reduction across the wavefront (GEBAL_BATCH <= warpSize).
    static_assert(BW <= 32, "GEBAL_BATCH must not exceed the wavefront size");
    if(tid < warpSize)
    {
        const int q = tid;
        bool any = false, nan = false;
        for(int jj = 0; jj < nb; jj++)
        {
            const I i = ib0 + jj;
            const gebal_part<S, I>& rc = red[0][jj];
            const gebal_part<S, I>& rr = red[1][jj];
            T a = 0, b = 0;
            gebal_vec_info<S, I> col, row;
            col.init(n);
            row.init(n);
            if(q < nb)
            {
                a = blk[q + jj * BW];
                b = blk[jj + q * BW];
                col.add(a, ib0 + q, true);
                row.add(b, ib0 + q, true);
            }
            col.wave_reduce();
            row.wave_reduce();
            col.combine(rc.amax, rc.iamax, rc.mod, rc.ssq, rc.unsafe);
            row.combine(rr.amax, rr.iamax, rr.mod, rr.ssq, rr.unsafe);
            S c = sqrt(col.ssq);
            S r = sqrt(row.ssq);

            // (col.unsafe and row.unsafe are the same on all the lanes)
            if(col.unsafe || row.unsafe)
            {
                gebal_scaled_ssq<S> cs, rs;
                cs.init();
                rs.init();
                cs.add(a);
                rs.add(b);
                cs.wave_reduce();
                rs.wave_reduce();
                cs.combine(rc.se, rc.sssq);
                rs.combine(rr.se, rr.sssq);
                if(col.unsafe)
                    c = cs.norm();
                if(row.unsafe)
                    r = rs.norm();
            }
            const S si = scale[i];

            // (f = 0: NaN, the balancing stops; f = 1: no scaling)
            const S f = gebal_factor(c, r, col.mod, row.mod, si);
            if(f == S(0))
            {
                nan = true;
                break;
            }
            if(f == S(1))
                continue;
            const S g = S(1) / f;
            any = true;

            // scale row i by g and column i by f in the block; A(i,i) by both, in the same
            // order as LAPACK
            if(q < nb)
            {
                if(q == jj)
                {
                    a *= g;
                    a *= f;
                    blk[jj + jj * BW] = a;
                }
                else
                {
                    b *= g;
                    a *= f;
                    blk[jj + q * BW] = b;
                    blk[q + jj * BW] = a;
                }
            }
            if(q == 0)
            {
                scale[i] = si * f;
                fac[jj] = f;
            }
            // (the next decision reads entries written by other lanes)
            __syncwarp();
        }
        if(q == 0)
        {
            st.any[bid] = any ? 1 : 0;
            if(any)
                st.noconv[bid] = 1;
            if(nan)
                st.stop[bid] = 1;
            s_any = any;
        }
    }
    __syncthreads();

    if(s_any)
        for(I idx = tid; idx < nb * nb; idx += BS)
            A[idx2D(ib0 + idx % nb, ib0 + idx / nb, lda)] = blk[(idx % nb) + (idx / nb) * BW];
}

/** GEBAL_MB_APPLY_KERNEL scales, for the batch ib0:ib1, the columns j by fac(j) over the
    rows 0:l (blockIdx.y = 0) and the rows j by 1/fac(j) over the columns k:n-1 (1),
    outside the diagonal block of the batch (segment p = blockIdx.x). **/
template <int BS, typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(BS) gebal_mb_apply_kernel(const I n,
                                                                  U AA,
                                                                  const rocblas_stride shiftA,
                                                                  const I lda,
                                                                  const rocblas_stride strideA,
                                                                  const I bc,
                                                                  char* ws,
                                                                  const I i0)
{
    constexpr int BW = GEBAL_BATCH;
    constexpr int NSEG = GEBAL_NSEG;
    gebal_mb_state<S, I> st(ws, n, bc);
    const I bid = hipBlockIdx_z;
    if(!st.any[bid])
        return;
    const I k = st.k[bid];
    const I l = st.l[bid];
    const I ib0 = std::max(i0, k);
    const I ib1 = std::min(i0 + BW - 1, l);
    const I nb = ib1 - ib0 + 1;
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    const S* fac = st.fac + rocblas_stride(bid) * BW;
    const I p = hipBlockIdx_x;
    const int tid = hipThreadIdx_x;

    if(hipBlockIdx_y == 0)
    {
        const int lane = tid % warpSize;
        const int wave = tid / warpSize;
        const int nwaves = BS / warpSize;
        const int64_t len = int64_t(l) + 1;
        const I r0 = I(p * len / NSEG);
        const I r1 = I((p + 1) * len / NSEG);
        for(int jj = wave; jj < nb; jj += nwaves)
        {
            const S f = fac[jj];
            if(f == S(1))
                continue;
            for(I r = r0 + lane; r < r1; r += warpSize)
                if(r < ib0 || r > ib1)
                    A[idx2D(r, ib0 + jj, lda)] *= f;
        }
    }
    else
    {
        constexpr int NCL = BS / BW;
        static_assert(BS % BW == 0, "the block size must be a multiple of GEBAL_BATCH");
        const int r = tid % BW;
        const int cl = tid / BW;
        const int64_t len = int64_t(n) - k;
        const I c0 = k + I(p * len / NSEG);
        const I c1 = k + I((p + 1) * len / NSEG);
        if(r < nb && fac[r] != S(1))
        {
            const S g = S(1) / fac[r];
            for(I c = c0 + cl; c < c1; c += NCL)
                if(c < ib0 || c > ib1)
                    A[idx2D(ib0 + r, c, lda)] *= g;
        }
    }
}

/** GEBAL_MB_FINAL_KERNEL sets ilo and ihi (ilo = ihi = 1 was set if all the eigenvalues
    were isolated). **/
template <typename I, typename S>
ROCSOLVER_KERNEL void gebal_mb_final_kernel(const I n, I* iloA, I* ihiA, const I bc, char* ws)
{
    gebal_mb_state<S, I> st(ws, n, bc);
    const I b = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(b < bc)
    {
        const bool all = (st.stop[b] == 2);
        iloA[b] = all ? 1 : st.k[b] + 1;
        ihiA[b] = all ? 1 : st.l[b] + 1;
    }
}

/** GEBAL_MULTIBLOCK balances the matrices of the batch with the multi-block path. It
    synchronizes the stream with the host after the permutation step and after each sweep
    of the scaling step. **/
template <typename T, typename I, typename S, typename U>
rocblas_status gebal_multiblock(rocblas_handle handle,
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
                                char* ws)
{
    constexpr int BS = GEBAL_BLOCKSIZE;
    constexpr int BW = GEBAL_BATCH;
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);
    const I bc = batch_count;
    const I bcb = (bc - 1) / BS1 + 1;
    gebal_mb_state<S, I> st(ws, n, bc);

    ROCSOLVER_LAUNCH_KERNEL((gebal_mb_init_kernel<S, I>), dim3(bcb), dim3(BS1), 0, stream, n, bc,
                            ws);
    if(job != rocsolver_balance_scale)
    {
        ROCSOLVER_LAUNCH_KERNEL((gebal_mb_rowcnt_kernel<BS, T, I, S>), dim3((n - 1) / 64 + 1, bc),
                                dim3(BS), 0, stream, n, A, shiftA, lda, strideA, bc, ws);
        ROCSOLVER_LAUNCH_KERNEL((gebal_mb_search_kernel<BS, T>), dim3(bc), dim3(BS), 0, stream,
                                true, n, A, shiftA, lda, strideA, scale, strideS, bc, ws);
        const I nwaves = BS / 64;
        ROCSOLVER_LAUNCH_KERNEL((gebal_mb_colcnt_kernel<BS, T, I, S>),
                                dim3(std::min((n - 1) / nwaves + 1, I(4096)), bc), dim3(BS), 0,
                                stream, n, A, shiftA, lda, strideA, bc, ws);
        ROCSOLVER_LAUNCH_KERNEL((gebal_mb_search_kernel<BS, T>), dim3(bc), dim3(BS), 0, stream,
                                false, n, A, shiftA, lda, strideA, scale, strideS, bc, ws);
    }
    ROCSOLVER_LAUNCH_KERNEL((gebal_mb_scale_init_kernel<T, I>), dim3((n - 1) / BS1 + 1, bc),
                            dim3(BS1), 0, stream, n, scale, strideS, bc, ws);

    if(job != rocsolver_balance_permute)
    {
        // active ranges, and the sweeps of the scaling step while a matrix is scaled
        std::vector<I> hk(bc), hl(bc), hstop(bc), hnoconv(bc);
        HIP_CHECK(hipMemcpyAsync(hk.data(), st.k, sizeof(I) * bc, hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipMemcpyAsync(hl.data(), st.l, sizeof(I) * bc, hipMemcpyDeviceToHost, stream));
        HIP_CHECK(
            hipMemcpyAsync(hstop.data(), st.stop, sizeof(I) * bc, hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        I kmin = n, lmax = -1;
        for(I b = 0; b < bc; b++)
            if(!hstop[b])
            {
                kmin = std::min(kmin, hk[b]);
                lmax = std::max(lmax, hl[b]);
            }

        bool active = (kmin <= lmax);
        while(active)
        {
            HIP_CHECK(hipMemsetAsync(st.noconv, 0, sizeof(I) * bc, stream));
            for(I i0 = kmin; i0 <= lmax; i0 += BW)
            {
                ROCSOLVER_LAUNCH_KERNEL((gebal_mb_stats_kernel<BS, T, I, S>), dim3(GEBAL_NSEG, 2, bc),
                                        dim3(BS), 0, stream, n, A, shiftA, lda, strideA, bc, ws, i0);
                ROCSOLVER_LAUNCH_KERNEL((gebal_mb_decide_kernel<BS, T>), dim3(bc), dim3(BS), 0,
                                        stream, n, A, shiftA, lda, strideA, scale, strideS, bc, ws,
                                        i0);
                ROCSOLVER_LAUNCH_KERNEL((gebal_mb_apply_kernel<BS, T, I, S>), dim3(GEBAL_NSEG, 2, bc),
                                        dim3(BS), 0, stream, n, A, shiftA, lda, strideA, bc, ws, i0);
            }
            HIP_CHECK(hipMemcpyAsync(hnoconv.data(), st.noconv, sizeof(I) * bc,
                                     hipMemcpyDeviceToHost, stream));
            HIP_CHECK(hipMemcpyAsync(hstop.data(), st.stop, sizeof(I) * bc, hipMemcpyDeviceToHost,
                                     stream));
            HIP_CHECK(hipStreamSynchronize(stream));

            // (a matrix that was not scaled in this sweep has converged)
            active = false;
            for(I b = 0; b < bc; b++)
            {
                if(!hstop[b] && !hnoconv[b])
                    hstop[b] = 3;
                active = active || !hstop[b];
            }
            if(active)
                HIP_CHECK(hipMemcpyAsync(st.stop, hstop.data(), sizeof(I) * bc,
                                         hipMemcpyHostToDevice, stream));
        }
    }

    ROCSOLVER_LAUNCH_KERNEL((gebal_mb_final_kernel<I, S>), dim3(bcb), dim3(BS1), 0, stream, n, ilo,
                            ihi, bc, ws);
    return rocblas_status_success;
}

template <typename T, typename I>
void rocsolver_gebal_getMemorySize(const rocsolver_balance job,
                                   const I n,
                                   const I batch_count,
                                   size_t* size_work)
{
    using S = decltype(std::real(T{}));
    // (multi-block path: see gebal_mb_layout; otherwise, the counts of non-zeros used by
    // the permutation step)
    if(n == 0 || batch_count == 0 || job == rocsolver_balance_none)
        *size_work = 0;
    else if(n >= GEBAL_MULTI_MIN)
        *size_work = gebal_mb_layout<S, I>(n, batch_count).size;
    else if(job == rocsolver_balance_scale)
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

    if(n >= GEBAL_MULTI_MIN && job != rocsolver_balance_none)
        return gebal_multiblock<T>(handle, job, n, A, shiftA, lda, strideA, ilo, ihi, scale,
                                   strideS, batch_count, reinterpret_cast<char*>(work));

    // (when n == 0 the kernel only sets ilo = 1 and ihi = 0)
    ROCSOLVER_LAUNCH_KERNEL((gebal_kernel<GEBAL_BLOCKSIZE, T>), dim3(batch_count),
                            dim3(GEBAL_BLOCKSIZE), 0, stream, job, n, A, shiftA, lda, strideA, ilo,
                            ihi, scale, strideS, work);

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE

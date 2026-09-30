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

#include "lapack/roclapack_trexc.hpp"
#include "rocauxiliary_lahqr.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

/*
 * ===========================================================================
 *    Device building blocks of the multishift QR algorithm with aggressive
 *    early deflation (LAPACK ZLAQR0, ZLAQR2 and ZLAQR5). All the routines are
 *    executed by a whole thread-block; indices are 1-based, as in LAPACK.
 * ===========================================================================
 */

/** HQR_BLOCK_SUM returns the sum of v over the thread-block to all the threads
    (in the same order in all of them). s_red is a shared buffer of 2 x BS/32 entries,
    used alternately (selected by buf), so that one barrier per call is enough. **/
template <int BS, typename S>
__host__ __device__ S hqr_block_sum(S v, S (*s_red)[HQR_RED(BS)], int& buf)
{
    static_assert(BS == 1 || BS % 64 == 0, "BS must be 1 or a multiple of 64");
#if defined(__HIP_DEVICE_COMPILE__)
    const int lane = hqr_tid() % warpSize;
    const int wave = hqr_tid() / warpSize;
    const int nwaves = BS / warpSize;

    for(int offset = warpSize / 2; offset > 0; offset /= 2)
        v += __shfl_xor(v, offset);
    if(lane == 0)
        s_red[buf][wave] = v;
    hqr_sync();
    v = s_red[buf][0];
    for(int w = 1; w < nwaves; w++)
        v += s_red[buf][w];
    buf = 1 - buf;
    return v;
#else
    return v;
#endif
}

/** HQR_NANMAX returns the maximum of a and b, or NaN if either of them is NaN. **/
template <typename S>
__host__ __device__ inline S hqr_nanmax(const S a, const S b)
{
    return (a != a) ? a : ((b != b) ? b : std::max(a, b));
}

/** HQR_BLOCK_MAXS returns the maximum of v (a non-negative real value, or NaN) over the
    thread-block to all the threads (NaN if any of the values is NaN), with the same
    buffering scheme as hqr_block_sum. **/
template <int BS, typename S>
__host__ __device__ S hqr_block_maxs(S v, S (*s_red)[HQR_RED(BS)], int& buf)
{
    static_assert(BS == 1 || BS % 64 == 0, "BS must be 1 or a multiple of 64");
#if defined(__HIP_DEVICE_COMPILE__)
    const int lane = hqr_tid() % warpSize;
    const int wave = hqr_tid() / warpSize;
    const int nwaves = BS / warpSize;

    for(int offset = warpSize / 2; offset > 0; offset /= 2)
        v = hqr_nanmax(v, S(__shfl_xor(v, offset)));
    if(lane == 0)
        s_red[buf][wave] = v;
    hqr_sync();
    v = s_red[buf][0];
    for(int w = 1; w < nwaves; w++)
        v = hqr_nanmax(v, s_red[buf][w]);
    buf = 1 - buf;
    return v;
#else
    return v;
#endif
}

/** HQR_BLOCK_NRM2 returns the 2-norm of the vector x of length m (stride 1) to all
    the threads. The entries are scaled by a power of two close to the largest one,
    so that the sum of squares cannot overflow or underflow. **/
template <int BS, typename T, typename I, typename S>
__host__ __device__ S hqr_block_nrm2(const I m, const T* x, S (*s_red)[HQR_RED(BS)], int& buf)
{
    const I tid = hqr_tid();
    // (a NaN entry makes the norm NaN, as in LAPACK DZNRM2)
    S amax = 0;
    for(I j = tid; j < m; j += BS)
        amax = hqr_nanmax(amax, hqr_nanmax(std::abs(x[j].real()), std::abs(x[j].imag())));
    amax = hqr_block_maxs<BS>(amax, s_red, buf);
    if(amax == 0 || !std::isfinite(amax))
        return amax;

    int e;
    frexp(amax, &e);
    S ssq = 0;
    for(I j = tid; j < m; j += BS)
    {
        S re = ldexp(x[j].real(), -e);
        S im = ldexp(x[j].imag(), -e);
        ssq += re * re + im * im;
    }
    ssq = hqr_block_sum<BS>(ssq, s_red, buf);
    return ldexp(std::sqrt(ssq), e);
}

/** HQR_BLOCK_LARFG generates an elementary reflector H of order m such that
    H^H * [alpha; x] = [beta; 0], with beta real (LAPACK ZLARFG with incx = 1).
    alpha is read from and written to *alpha (beta on exit), and x (m-1 entries,
    stride 1) is overwritten with the tail of the Householder vector. tau is returned
    to all the threads. **/
template <int BS, typename T, typename I, typename S>
__host__ __device__ T hqr_block_larfg(const I m, T* alpha, T* x, S (*s_red)[HQR_RED(BS)], int& buf)
{
    const I tid = hqr_tid();

    if(m <= 0)
        return T(0);

    S xnorm = hqr_block_nrm2<BS>(m - 1, x, s_red, buf);
    T a = *alpha;
    S alphr = a.real();
    S alphi = a.imag();

    if(xnorm == 0 && alphi == 0)
    {
        // H = I (all the threads have read alpha before the caller changes it)
        hqr_sync();
        return T(0);
    }

    // general case
    S beta = -std::copysign(hqr_lapy3(alphr, alphi, xnorm), alphr);
    const S safmin = hqr_safmin<S>() / hqr_eps<S>();
    const S rsafmn = S(1) / safmin;

    int knt = 0;
    if(std::abs(beta) < safmin)
    {
        // xnorm, beta may be inaccurate; scale x and recompute them
        do
        {
            knt++;
            hqr_sync();
            for(I j = tid; j < m - 1; j += BS)
                x[j] = rsafmn * x[j];
            beta = beta * rsafmn;
            alphi = alphi * rsafmn;
            alphr = alphr * rsafmn;
        } while(std::abs(beta) < safmin && knt < 20);
        hqr_sync();

        // new beta is at most 1, at least safmin
        xnorm = hqr_block_nrm2<BS>(m - 1, x, s_red, buf);
        a = T(alphr, alphi);
        beta = -std::copysign(hqr_lapy3(alphr, alphi, xnorm), alphr);
    }
    T tau = T((beta - alphr) / beta, -alphi / beta);
    T scal = hqr_zladiv(T(1), a - beta);
    for(I j = tid; j < m - 1; j += BS)
        x[j] = scal * x[j];

    // if alpha is subnormal, it may lose relative accuracy
    for(int j = 0; j < knt; j++)
        beta = beta * safmin;

    // all the threads have read *alpha
    hqr_sync();
    if(tid == 0)
        *alpha = T(beta);
    hqr_sync();
    return tau;
}

/** HQR_BLOCK_LARF_LEFT applies H = I - tau * v * v^H from the left to the m-by-n
    matrix C: C <- H * C. v has m entries (stride 1). Each thread processes whole columns. **/
template <int BS, typename T, typename I>
__host__ __device__ void
    hqr_block_larf_left(const I m, const I n, const T* v, const T tau, T* C, const I ldc)
{
    const I tid = hqr_tid();
    if(tau == T(0))
        return;
    for(I j = tid; j < n; j += BS)
    {
        T w = 0;
        for(I i = 0; i < m; i++)
            w += conj(v[i]) * C[idx2D(i, j, ldc)];
        w = tau * w;
        for(I i = 0; i < m; i++)
            C[idx2D(i, j, ldc)] -= v[i] * w;
    }
}

/** HQR_BLOCK_LARF_RIGHT applies H = I - tau * v * v^H from the right to the m-by-n
    matrix C: C <- C * H. v has n entries (stride 1). Each thread processes whole rows. **/
template <int BS, typename T, typename I>
__host__ __device__ void
    hqr_block_larf_right(const I m, const I n, const T* v, const T tau, T* C, const I ldc)
{
    const I tid = hqr_tid();
    if(tau == T(0))
        return;
#if !defined(__HIP_DEVICE_COMPILE__)
    if constexpr(BS == 1)
    {
        // (on the host, by columns, which are contiguous: w = tau * C * v, then
        // C <- C - w * v^H)
        thread_local std::vector<T> w;
        w.assign(m, T(0));
        for(I j = 0; j < n; j++)
        {
            const T vj = v[j];
            for(I i = 0; i < m; i++)
                w[i] += C[idx2D(i, j, ldc)] * vj;
        }
        for(I i = 0; i < m; i++)
            w[i] = tau * w[i];
        for(I j = 0; j < n; j++)
        {
            const T vj = conj(v[j]);
            for(I i = 0; i < m; i++)
                C[idx2D(i, j, ldc)] -= w[i] * vj;
        }
        return;
    }
#endif
    for(I i = tid; i < m; i += BS)
    {
        T w = 0;
        for(I j = 0; j < n; j++)
            w += C[idx2D(i, j, ldc)] * v[j];
        w = tau * w;
        for(I j = 0; j < n; j++)
            C[idx2D(i, j, ldc)] -= w * conj(v[j]);
    }
}

/*
 * ===========================================================================
 *    Aggressive early deflation (LAPACK ZLAQR2), in three parts:
 *    - aed_setup_block (device): selects the window and copies it to T (V = I),
 *    - aed_core_block (device, or host with BS = 1): Schur form of the window,
 *      deflation tests, reordering, and return to Hessenberg form,
 *    - aed_finish_block (device): copies the window back into H.
 *    The off-window updates (H(ltop:kwtop-1, kwtop:kbot) <- H * V, H(kwtop:kbot,
 *    kbot+1:n) <- V^H * H if wantt, and Z(iloz:ihiz, kwtop:kbot) <- Z * V if wantz)
 *    are left to the caller, when update is true.
 * ===========================================================================
 */

/** AED_SETUP_BLOCK selects the deflation window (jw, kwtop) and the spike s. If
    the window is empty or 1-by-1, it completes the deflation (ns, nd) and returns
    false. Otherwise, it copies the upper Hessenberg part of the window to T (zero
    below), sets V = I, and returns true. **/
template <int BS, typename T, typename I>
__device__ bool aed_setup_block(const I n,
                                const I ktop,
                                const I kbot,
                                const I nw,
                                T* H,
                                const I ldh,
                                T* W,
                                T* Tw,
                                const I ldt,
                                T* V,
                                const I ldv,
                                I& jw,
                                I& kwtop,
                                T& s,
                                I& ns,
                                I& nd)
{
    using S = decltype(std::real(T{}));

    const I tid = hqr_tid();
    auto h = [&](const I i, const I j) -> T& { return H[idx2D(i - 1, j - 1, ldh)]; };
    auto t = [&](const I i, const I j) -> T& { return Tw[idx2D(i - 1, j - 1, ldt)]; };
    auto v = [&](const I i, const I j) -> T& { return V[idx2D(i - 1, j - 1, ldv)]; };

    const S safmin = hqr_safmin<S>();
    const S ulp = hqr_ulp<S>();
    const S smlnum = safmin * (S(n) / ulp);

    // nothing to do for an empty active block or window
    if(ktop > kbot || nw < 1)
    {
        jw = 0;
        kwtop = kbot + 1;
        s = T(0);
        ns = 0;
        nd = 0;
        return false;
    }

    // setup deflation window
    jw = std::min(nw, kbot - ktop + 1);
    kwtop = kbot - jw + 1;
    s = (kwtop == ktop) ? T(0) : h(kwtop, kwtop - 1);

    if(kbot == kwtop)
    {
        // 1-by-1 deflation window: not much to do
        ns = 1;
        nd = 0;
        T hkk = h(kwtop, kwtop);
        if(hqr_cabs1(s) <= std::max(smlnum, ulp * hqr_cabs1(hkk)))
        {
            ns = 0;
            nd = 1;
        }
        __syncthreads();
        if(tid == 0)
        {
            W[kwtop - 1] = hkk;
            if(ns == 0 && kwtop > ktop)
                h(kwtop, kwtop - 1) = T(0);
        }
        __syncthreads();
        return false;
    }

    // convert to spike-triangular form: T = window (upper Hessenberg part), V = I
    for(I j = 1; j <= jw; j++)
        for(I i = 1 + tid; i <= jw; i += BS)
        {
            t(i, j) = (i <= j + 1) ? h(kwtop + i - 1, kwtop + j - 1) : T(0);
            v(i, j) = (i == j) ? T(1) : T(0);
        }
    __syncthreads();
    return true;
}

/** HSEQR_IPARMQ returns the number of shifts (ISPEC = 15) and the deflation window
    size (ISPEC = 13) recommended by LAPACK IPARMQ for an active block of order nh. **/
template <typename I>
__host__ __device__ void hseqr_iparmq(const I nh, I& ns, I& nw)
{
    ns = 2;
    if(nh >= 30)
        ns = 4;
    if(nh >= 60)
        ns = 10;
    if(nh >= 150)
        ns = std::max(I(10), I(nh / I(::lround(::log(double(nh)) / ::log(2.0)))));
    if(nh >= 590)
        ns = 64;
    if(nh >= 3000)
        ns = 128;
    if(nh >= 6000)
        ns = 256;
    ns = std::max(I(2), ns - ns % 2);
    nw = (nh <= 500) ? ns : 3 * ns / 2;
}

/** Maximum number of shifts of the sweeps of laqr4_block, and the size of the window
    above which aed_core_block uses laqr4_block for its Schur form (on the device). **/
#define LAQR4_MAX_SHIFTS 32
#define LAQR4_NMIN 64

template <int BS, typename T, typename I>
__device__ I laqr4_block(const I n,
                         T* H,
                         const I ldh,
                         T* W,
                         T* Z,
                         const I ldz,
                         I* st,
                         T* stT,
                         T (*sV)[3],
                         I (*s_ired)[HQR_RED(BS)],
                         decltype(std::real(T{})) (*s_sred)[HQR_RED(BS)],
                         int& ibuf,
                         int& sbuf,
                         T* lds_ws);

/** HQR_HOST_OPS holds host versions of the routines that aed_core_block uses (BS = 1),
    which may be compiled for other instruction sets than the rest of the library (see
    hseqr_host_ops); the generic routines are used for the null members. **/
template <typename T, typename I>
struct hqr_host_ops
{
    // Schur form and vectors of the n-by-n T (V = I on entry), as lahqr_block; returns info
    I (*schur)(I n, T* Tw, I ldt, T* W, T* V, I ldv);
    // trexc_block (with wantq)
    void (*trexc)(I n, T* A, I ldt, T* Q, I ldq, I ifst, I ilst);
    // hqr_block_larf_left and hqr_block_larf_right
    void (*larf_left)(I m, I n, const T* v, T tau, T* C, I ldc);
    void (*larf_right)(I m, I n, const T* v, T tau, T* C, I ldc);
};

/** AED_CORE_BLOCK performs the part of ZLAQR2 that only involves the jw-by-jw
    window (jw > 1): given the spike s, T (the window in upper Hessenberg form, zero
    below) and V = I, it computes the Schur form of T (ZLAHQR), tests the spike for
    deflation from the bottom (moving undeflatable eigenvalues up with TREXC),
    sorts the undeflated eigenvalues, and, if needed, reflects the spike back and
    returns T to Hessenberg form (ZGEHRD + ZUNMHR). The eigenvalues are returned in
    Wsh (jw entries), work has at least jw entries, and on exit:
    - ns: number of undeflated eigenvalues (the shifts),
    - nd: number of deflated eigenvalues,
    - update: whether H must be updated with T and V,
    - spike: the new value of H(kwtop, kwtop-1).
    On the device it is executed by all the threads of a thread-block of BS threads;
    on the host, with BS = 1. On the device, lds_ws (shared memory of HQR_LDS_WS_SIZE
    entries, if given, with BS >= HQR_LDS_NMAX) is used for the Schur form of windows of
    at most HQR_LDS_NMAX entries (lahqr_lds_block). On the host, the routines of host_ops (if
    given) replace the generic ones (see hqr_host_ops). With MULTISHIFT (on the device, with st4, stT4 and sV4, the
    status arrays and the reflections of laqr4_block in shared memory), windows with
    jw > LAQR4_NMIN are reduced to Schur form by laqr4_block (the multishift QR
    algorithm, as LAPACK ZLAQR3 calls ZLAQR4), whose own deflation windows use this
    function without MULTISHIFT. (It is a template parameter so that the kernels of small
    windows do not contain laqr4_block.) **/
template <int BS, typename T, typename I, bool MULTISHIFT = false>
__host__ __device__ void aed_core_block(const I n,
                                        const I jw,
                                        T s,
                                        T* Tw,
                                        const I ldt,
                                        T* V,
                                        const I ldv,
                                        T* work,
                                        T* Wsh,
                                        I& ns_out,
                                        I& nd_out,
                                        bool& update,
                                        T& spike,
                                        I (*s_ired)[HQR_RED(BS)],
                                        decltype(std::real(T{})) (*s_sred)[HQR_RED(BS)],
                                        int& ibuf,
                                        int& sbuf,
                                        T* lds_ws = nullptr,
                                        I* st4 = nullptr,
                                        T* stT4 = nullptr,
                                        T (*sV4)[3] = nullptr,
                                        const hqr_host_ops<T, I>* host_ops = nullptr)
{
    using S = decltype(std::real(T{}));

    const I tid = hqr_tid();
    auto t = [&](const I i, const I j) -> T& { return Tw[idx2D(i - 1, j - 1, ldt)]; };
    auto v = [&](const I i, const I j) -> T& { return V[idx2D(i - 1, j - 1, ldv)]; };

    // (on the host, the routines of host_ops, if given, replace the generic ones)
    auto trexc = [&](const I ifst, const I ilst) {
#if !defined(__HIP_DEVICE_COMPILE__)
        if(host_ops && host_ops->trexc)
            return host_ops->trexc(jw, Tw, ldt, V, ldv, ifst, ilst);
#endif
        trexc_block<BS>(true, jw, Tw, ldt, V, ldv, ifst, ilst);
    };
    auto larf_left = [&](const I m, const I nc, const T* x, const T tau, T* C, const I ldc) {
#if !defined(__HIP_DEVICE_COMPILE__)
        if(host_ops && host_ops->larf_left)
            return host_ops->larf_left(m, nc, x, tau, C, ldc);
#endif
        hqr_block_larf_left<BS>(m, nc, x, tau, C, ldc);
    };
    auto larf_right = [&](const I m, const I nc, const T* x, const T tau, T* C, const I ldc) {
#if !defined(__HIP_DEVICE_COMPILE__)
        if(host_ops && host_ops->larf_right)
            return host_ops->larf_right(m, nc, x, tau, C, ldc);
#endif
        hqr_block_larf_right<BS>(m, nc, x, tau, C, ldc);
    };

    const S safmin = hqr_safmin<S>();
    const S ulp = hqr_ulp<S>();
    const S smlnum = safmin * (S(n) / ulp);

    I infqr = 0;
#if defined(__HIP_DEVICE_COMPILE__)
    bool multishift = false;
    if constexpr(MULTISHIFT)
        multishift = (jw > LAQR4_NMIN);
    if(multishift)
    {
        // large windows: multishift QR (it uses the part of T below its first subdiagonal
        // as workspace, whose converged entries it sets to zero)
        if constexpr(MULTISHIFT)
            infqr = laqr4_block<BS>(jw, Tw, ldt, Wsh, V, ldv, st4, stT4, sV4, s_ired, s_sred, ibuf,
                                    sbuf, lds_ws);
        hqr_sync();
        for(I j = 1; j <= jw - 2; j++)
            for(I i = j + 2 + tid; i <= jw; i += BS)
                t(i, j) = T(0);
    }
    // small windows: Schur form in shared memory (see lahqr_lds_block)
    else if constexpr(BS >= HQR_LDS_NMAX)
    {
        if(lds_ws && jw <= HQR_LDS_NMAX)
            infqr = lahqr_lds_block<BS>(jw, Tw, ldt, Wsh, V, ldv, lds_ws);
        else
            infqr = lahqr_block<BS>(true, true, jw, I(1), jw, Tw, ldt, Wsh, I(1), jw, V, ldv,
                                    s_ired, ibuf);
    }
    else
        infqr = lahqr_block<BS>(true, true, jw, I(1), jw, Tw, ldt, Wsh, I(1), jw, V, ldv, s_ired,
                                ibuf);
#else
    infqr = (host_ops && host_ops->schur)
        ? host_ops->schur(jw, Tw, ldt, Wsh, V, ldv)
        : lahqr_block<BS>(true, true, jw, I(1), jw, Tw, ldt, Wsh, I(1), jw, V, ldv, s_ired, ibuf);
#endif
    hqr_sync();

    // deflation detection loop
    I ns = jw;
    I ilst = infqr + 1;
    for(I knt = infqr + 1; knt <= jw; knt++)
    {
        // small spike tip deflation test
        S foo = hqr_cabs1(t(ns, ns));
        if(foo == 0)
            foo = hqr_cabs1(s);
        if(hqr_cabs1(s) * hqr_cabs1(v(1, ns)) <= std::max(smlnum, ulp * foo))
        {
            // one more converged eigenvalue
            ns = ns - 1;
        }
        else
        {
            // one undeflatable eigenvalue; move it up out of the way
            hqr_sync();
            trexc(ns, ilst);
            ilst = ilst + 1;
        }
    }

    // return to Hessenberg form
    if(ns == 0)
        s = T(0);

    if(ns < jw)
    {
        // sorting the diagonal of T improves accuracy for graded matrices
        for(I i = infqr + 1; i <= ns; i++)
        {
            I ifst = i;
            for(I j = i + 1; j <= ns; j++)
                if(hqr_cabs1(t(j, j)) > hqr_cabs1(t(ifst, ifst)))
                    ifst = j;
            if(ifst != i)
            {
                hqr_sync();
                trexc(ifst, i);
            }
        }
    }
    hqr_sync();

    // restore shift/eigenvalue array from T
    for(I i = infqr + 1 + tid; i <= jw; i += BS)
        Wsh[i - 1] = t(i, i);

    update = (ns < jw || s == T(0));
    if(update && ns > 1 && s != T(0))
    {
        // reflect spike back into lower triangle
        for(I i = 1 + tid; i <= ns; i += BS)
            work[i - 1] = conj(v(1, i));
        hqr_sync();
        T tau = hqr_block_larfg<BS>(ns, work, work + 1, s_sred, sbuf);
        if(tid == 0)
            work[0] = T(1);

        // zero T below the first subdiagonal
        for(I j = 1; j <= jw - 2; j++)
            for(I i = j + 2 + tid; i <= jw; i += BS)
                t(i, j) = T(0);
        hqr_sync();

        larf_left(ns, jw, work, conj(tau), Tw, ldt);
        hqr_sync();
        larf_right(ns, ns, work, tau, Tw, ldt);
        larf_right(jw, ns, work, tau, V, ldv);
        hqr_sync();

        // Hessenberg reduction of T(1:ns, 1:ns) (ZGEHRD with ilo = 1 and ihi = ns),
        // applying the reflectors also to V (ZUNMHR)
        for(I i = 1; i <= ns - 1; i++)
        {
            // compute the elementary reflector H(i) to annihilate T(i+2:ns, i)
            T taui = hqr_block_larfg<BS>(ns - i, &t(i + 1, i), &t(std::min(i + 2, jw), i), s_sred,
                                         sbuf);
            T alpha = t(i + 1, i);
            hqr_sync();
            if(tid == 0)
                t(i + 1, i) = T(1);
            hqr_sync();

            // apply H(i) to T(1:ns, i+1:ns) from the right, to T(i+1:ns, i+1:jw) from the
            // left, and to V(1:jw, i+1:ns) from the right
            larf_right(ns, ns - i, &t(i + 1, i), taui, &t(1, i + 1), ldt);
            hqr_sync();
            larf_left(ns - i, jw - i, &t(i + 1, i), conj(taui), &t(i + 1, i + 1), ldt);
            larf_right(jw, ns - i, &t(i + 1, i), taui, &v(1, i + 1), ldv);
            hqr_sync();
            if(tid == 0)
                t(i + 1, i) = alpha;
            hqr_sync();
        }
    }

    spike = s * conj(v(1, 1));
    ns_out = ns - infqr;
    nd_out = jw - ns;
    hqr_sync();
}

/** AED_FINISH_BLOCK copies the window T back into H(kwtop:kwtop+jw-1, ...) and sets
    H(kwtop, kwtop-1) = spike, if update is true. **/
template <int BS, typename T, typename I>
__device__ void aed_finish_block(const I kwtop,
                                 const I jw,
                                 T* H,
                                 const I ldh,
                                 const T* Tw,
                                 const I ldt,
                                 const bool update,
                                 const T spike)
{
    const I tid = hqr_tid();
    if(update)
    {
        for(I j = 1; j <= jw; j++)
            for(I i = 1 + tid; i <= std::min(j + 1, jw); i += BS)
                H[idx2D(kwtop + i - 2, kwtop + j - 2, ldh)] = Tw[idx2D(i - 1, j - 1, ldt)];
        __syncthreads();
        if(tid == 0 && kwtop > 1)
            H[idx2D(kwtop - 1, kwtop - 2, ldh)] = spike;
    }
    __syncthreads();
}

/** HQR_LAQR1 computes a scalar multiple of the first column of
    (H - s1*I)*(H - s2*I) for the n-by-n (n = 2 or 3) leading block of H
    (LAPACK ZLAQR1). Executed by a single thread. **/
template <typename T, typename I>
__device__ void hqr_laqr1(const int nn, const T* H, const I ldh, const T s1, const T s2, T* v)
{
    using S = decltype(std::real(T{}));
    auto h = [&](const I i, const I j) -> T { return H[idx2D(i - 1, j - 1, ldh)]; };

    if(nn == 2)
    {
        S s = hqr_cabs1(h(1, 1) - s2) + hqr_cabs1(h(2, 1));
        if(s == 0)
        {
            v[0] = T(0);
            v[1] = T(0);
        }
        else
        {
            T h21s = h(2, 1) / s;
            v[0] = h21s * h(1, 2) + (h(1, 1) - s1) * ((h(1, 1) - s2) / s);
            v[1] = h21s * (h(1, 1) + h(2, 2) - s1 - s2);
        }
    }
    else
    {
        S s = hqr_cabs1(h(1, 1) - s2) + hqr_cabs1(h(2, 1)) + hqr_cabs1(h(3, 1));
        if(s == 0)
        {
            v[0] = T(0);
            v[1] = T(0);
            v[2] = T(0);
        }
        else
        {
            T h21s = h(2, 1) / s;
            T h31s = h(3, 1) / s;
            v[0] = (h(1, 1) - s1) * ((h(1, 1) - s2) / s) + h(1, 2) * h21s;
            v[1] = h21s * (h(1, 1) + h(2, 2) - s1 - s2);
            v[2] = h31s * (h(1, 1) + h(3, 3) - s1 - s2) + h21s * h(3, 2);
            // (h(3,1) is zero when a bulge is introduced at the top of the active block; then
            // h(1,3) and h(2,3) are not read, as the other thread-blocks of the chase may be
            // updating them, see laqr5_chunk_block)
            if(h31s != T(0))
            {
                v[0] += h(1, 3) * h31s;
                v[1] += h(2, 3) * h31s;
            }
        }
    }
}

/** LAQR5_GRID_BARRIER synchronizes the G thread-blocks of a grid (all of them resident),
    and makes the global memory writes of each one visible to the others. bar points to
    an arrivals counter, which must be 0 when the kernel starts (bar[1] is unused): it
    grows by G per barrier, and each thread-block waits until it reaches the next multiple
    of G. Each thread-block releases its writes
    (a release fence) before it arrives, and acquires those of the others (an acquire fence,
    which only invalidates the non-coherent cached data) after the last one arrives: a
    full fence would also write back the L2 cache (on gfx94x, of its XCD), which is slow
    when it holds many modified lines (for example of the matrix products on the side
    stream). **/
__device__ inline void laqr5_grid_barrier(unsigned* bar, const unsigned G)
{
    // (on gfx94x each XCD has its own L2 cache: every wavefront waits for its global memory
    // writes to reach it before the thread-block arrives, so that the write-back of the L2
    // by the release fence of thread 0 includes them; a barrier of the thread-block alone
    // does not wait for them)
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__)
    __builtin_amdgcn_s_waitcnt(0x0F70); // vmcnt(0)
#endif
    __syncthreads();
    if(hipThreadIdx_x == 0)
    {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__)
        // (the arrival must follow the completion of the write-back of the release fence;
        // the compiler does not always wait for it before the atomic)
        __builtin_amdgcn_s_waitcnt(0x0F70); // vmcnt(0)
#endif
        const unsigned old = __hip_atomic_fetch_add(bar, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        const unsigned target = old - old % G + G;
        while(int(__hip_atomic_load(bar, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) - target) < 0)
            __builtin_amdgcn_s_sleep(1);
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    }
    __syncthreads();
}

/*
 * ===========================================================================
 *    LAQR5_CHUNK_BLOCK: near-the-diagonal part of one chunk of a small-bulge
 *    multishift QR sweep.
 *
 *    The sweep follows LAPACK 3.9.0 ZLAQR5, in which the 2-shift bulges are
 *    spaced 3 rows apart (LAPACK 3.10 and later pack them 2 rows apart). With
 *    this spacing, the reflections of different bulges act on disjoint rows and
 *    columns, so that each step of the chase can be applied in parallel across
 *    bulges (a fixed number of barriers per step, independently of the number of
 *    shifts), whereas the tightly packed version requires the bulges to be processed
 *    one after the other. The deflation checks and the handling of collapsed bulges
 *    are those of LAPACK 3.9.0.
 *
 *    Without accum, the reflections are applied to the full rows and columns of H and,
 *    if wantz, to Z. With accum (KACC22 = 1; the 2-by-2 block structured update of
 *    KACC22 = 2 is not used), only the window of the chunk is updated, and the
 *    reflections of each step krcol are stored in Vbuf (3*(nbmps+1) entries per step:
 *    the 3 entries of the reflection of each bulge), from which the caller forms U
 *    (laqr5_build_u_block) for the far-from-diagonal updates.
 *
 *    sh contains the shifts (sh[0:2*nbmps-1]), and sV is shared workspace for
 *    nbmps+1 reflections.
 *
 *    With accum, the chunk may be chased by G > 1 thread-blocks (w = 0:G-1; bar
 *    are the counters of a laqr5_grid_barrier). Thread-block 0 generates the
 *    reflections of each step (stored in Vbuf, from which the others read them),
 *    multiplies the columns k+1:k+6 of each bulge by them from the left, and the
 *    rows k:k+3 from the right: these contain all the entries that the deflation
 *    checks, the fill-in of the last rows and the reflections of the next step use,
 *    and they depend on no other part of the multiplications. The others multiply
 *    the columns to the right from the left, then the rows above from the right. Of
 *    the latter, the rows k'+1:k'+3 of an upper bulge m' <= m-2 in the columns k+1:k+3
 *    of bulge m form a 3-by-3 block that only the reflections of m' (from the left) and
 *    of m (from the right) touch: one thread applies both. Thread-block 0 also
 *    multiplies the rows k-2:k-1 of bulge m-1 from the right (it multiplied them from
 *    the left), and the rows above the first bulge are only multiplied from the right.
 *    Thus there is one grid barrier per step, after the reflections are generated.
 * ===========================================================================
 */
template <int BS, typename T, typename I>
__device__ void laqr5_chunk_block(const bool wantt,
                                  const bool wantz,
                                  const bool accum,
                                  const I n,
                                  const I ktop,
                                  const I kbot,
                                  const I nbmps,
                                  const I incol,
                                  const T* sh,
                                  T* H,
                                  const I ldh,
                                  const I iloz,
                                  const I ihiz,
                                  T* Z,
                                  const I ldz,
                                  T* Vbuf,
                                  T (*sV)[3],
                                  const I G = 1,
                                  const I w = 0,
                                  unsigned* bar = nullptr)
{
    using S = decltype(std::real(T{}));

    const I tid = hipThreadIdx_x;
    auto h = [&](const I i, const I j) -> T& { return H[idx2D(i - 1, j - 1, ldh)]; };
    auto z = [&](const I i, const I j) -> T& { return Z[idx2D(i - 1, j - 1, ldz)]; };
    auto vv = [&](const I r, const I m) -> T& { return sV[m][r - 1]; }; // V(r, m), 1-based
    auto s = [&](const I i) -> T { return sh[i - 1]; }; // S(i), 1-based

    const S safmin = hqr_safmin<S>();
    const S ulp = hqr_ulp<S>();
    const S smlnum = safmin * (S(n) / ulp);

    const I kdu = 6 * nbmps - 3;
    const I ndcol = incol + kdu;

    // (with accum, the reflections of each step are stored in Vbuf, and U is formed
    // afterwards by laqr5_build_u_block)
    const I ldvb = 3 * (nbmps + 1);

    // test for a negligible subdiagonal entry h(k+1,k) (Ahues & Tisseur)
    auto vigilant = [&](const I k) {
        if(h(k + 1, k) != T(0))
        {
            S tst1 = hqr_cabs1(h(k, k)) + hqr_cabs1(h(k + 1, k + 1));
            if(tst1 == 0)
            {
                if(k >= ktop + 1)
                    tst1 = tst1 + hqr_cabs1(h(k, k - 1));
                if(k >= ktop + 2)
                    tst1 = tst1 + hqr_cabs1(h(k, k - 2));
                if(k >= ktop + 3)
                    tst1 = tst1 + hqr_cabs1(h(k, k - 3));
                if(k <= kbot - 2)
                    tst1 = tst1 + hqr_cabs1(h(k + 2, k + 1));
                if(k <= kbot - 3)
                    tst1 = tst1 + hqr_cabs1(h(k + 3, k + 1));
                if(k <= kbot - 4)
                    tst1 = tst1 + hqr_cabs1(h(k + 4, k + 1));
            }
            if(hqr_cabs1(h(k + 1, k)) <= std::max(smlnum, ulp * tst1))
            {
                S h12 = std::max(hqr_cabs1(h(k + 1, k)), hqr_cabs1(h(k, k + 1)));
                S h21 = std::min(hqr_cabs1(h(k + 1, k)), hqr_cabs1(h(k, k + 1)));
                S h11 = std::max(hqr_cabs1(h(k + 1, k + 1)), hqr_cabs1(h(k, k) - h(k + 1, k + 1)));
                S h22 = std::min(hqr_cabs1(h(k + 1, k + 1)), hqr_cabs1(h(k, k) - h(k + 1, k + 1)));
                S scl = h11 + h12;
                S tst2 = h22 * (h11 / scl);
                if(tst2 == 0 || h21 * (h12 / scl) <= std::max(smlnum, ulp * tst2))
                    h(k + 1, k) = T(0);
            }
        }
    };

    // (thread-block 0 leads: it generates the reflections, and does the deflation checks
    // and the fill-in)
    const bool lead = (w == 0);

    const I krlast = std::min(incol + 3 * nbmps - 3, kbot - 2);
    for(I krcol = incol; krcol <= krlast; krcol++)
    {
        // Bulges number mtop to mbot are active double implicit shift bulges. There may
        // or may not also be a small 2-by-2 bulge, if there is room. (Fortran integer
        // division truncates toward zero, as in C++.)
        const I mtop = std::max(I(1), ((ktop - 1) - krcol + 2) / 3 + 1);
        const I mbot = std::min(nbmps, (kbot - krcol) / 3);
        const I m22 = mbot + 1;
        const bool bmp22 = (mbot < nbmps) && (krcol + 3 * (m22 - 1) == kbot - 2);

        // 1. generate reflections to chase the chain right one column (one thread per
        //    bulge; the bulges read and write disjoint entries of H)
        for(I m = mtop + tid; lead && m <= mbot; m += BS)
        {
            const I k = krcol + 3 * (m - 1);
            T v3[3];
            if(k == ktop - 1)
            {
                hqr_laqr1(3, &h(ktop, ktop), ldh, s(2 * m - 1), s(2 * m), v3);
                T alpha = v3[0];
                T tau;
                hqr_larfg<3>(alpha, v3 + 1, tau);
                v3[0] = tau;
            }
            else
            {
                T beta = h(k + 1, k);
                v3[1] = h(k + 2, k);
                v3[2] = h(k + 3, k);
                T tau;
                hqr_larfg<3>(beta, v3 + 1, tau);
                v3[0] = tau;

                // A bulge may collapse because of vigilant deflation or destructive
                // underflow. In the underflow case, try the two-small-subdiagonals trick
                // to try to reinflate the bulge.
                if(h(k + 3, k) != T(0) || h(k + 3, k + 1) != T(0) || h(k + 3, k + 2) == T(0))
                {
                    // typical case: not collapsed (yet)
                    h(k + 1, k) = beta;
                    h(k + 2, k) = T(0);
                    h(k + 3, k) = T(0);
                }
                else
                {
                    // atypical case: collapsed. Attempt to reintroduce ignoring h(k+1,k)
                    // and h(k+2,k). If the fill resulting from the new reflector is too
                    // large, then abandon it. Otherwise, use the new one.
                    T vt[3];
                    hqr_laqr1(3, &h(k + 1, k + 1), ldh, s(2 * m - 1), s(2 * m), vt);
                    T alpha = vt[0];
                    T taut;
                    hqr_larfg<3>(alpha, vt + 1, taut);
                    vt[0] = taut;
                    T refsum = conj(vt[0]) * (h(k + 1, k) + conj(vt[1]) * h(k + 2, k));

                    if(hqr_cabs1(h(k + 2, k) - refsum * vt[1]) + hqr_cabs1(refsum * vt[2]) > ulp
                           * (hqr_cabs1(h(k, k)) + hqr_cabs1(h(k + 1, k + 1))
                              + hqr_cabs1(h(k + 2, k + 2))))
                    {
                        // starting a new bulge here would create non-negligible fill;
                        // use the old one with trepidation
                        h(k + 1, k) = beta;
                        h(k + 2, k) = T(0);
                        h(k + 3, k) = T(0);
                    }
                    else
                    {
                        // starting a new bulge here would create only negligible fill;
                        // replace the old reflector with the new one
                        h(k + 1, k) = h(k + 1, k) - refsum;
                        h(k + 2, k) = T(0);
                        h(k + 3, k) = T(0);
                        v3[0] = vt[0];
                        v3[1] = vt[1];
                        v3[2] = vt[2];
                    }
                }
            }
            vv(1, m) = v3[0];
            vv(2, m) = v3[1];
            vv(3, m) = v3[2];
        }

        // generate a 2-by-2 reflection, if needed
        if(lead && bmp22 && tid == BS - 1)
        {
            const I k = krcol + 3 * (m22 - 1);
            T v2[2];
            if(k == ktop - 1)
            {
                hqr_laqr1(2, &h(k + 1, k + 1), ldh, s(2 * m22 - 1), s(2 * m22), v2);
                T beta = v2[0];
                T tau;
                hqr_larfg<2>(beta, v2 + 1, tau);
                v2[0] = tau;
            }
            else
            {
                T beta = h(k + 1, k);
                v2[1] = h(k + 2, k);
                T tau;
                hqr_larfg<2>(beta, v2 + 1, tau);
                v2[0] = tau;
                h(k + 1, k) = beta;
                h(k + 2, k) = T(0);
            }
            vv(1, m22) = v2[0];
            vv(2, m22) = v2[1];
            vv(3, m22) = T(0);
        }
        __syncthreads();

        // with accum, store the reflections of this step (U is formed later from them, Z
        // is updated with a matrix-matrix multiply, and the other thread-blocks read them)
        const I mlast = mbot + (bmp22 ? 1 : 0);
        const I nb = mlast - mtop + 1;
        // (Vbuf is only given with accum, which G > 1 requires)
        T* vb = accum ? Vbuf + (krcol - incol) * ldvb : nullptr;
        if(accum && lead)
        {
            for(I idx = tid; idx < 3 * nb; idx += BS)
            {
                const I m = mtop + idx / 3;
                const I r = 1 + idx % 3;
                vb[3 * (m - 1) + r - 1] = vv(r, m);
            }
        }
        if(G > 1)
        {
            laqr5_grid_barrier(bar, G);
            if(!lead)
            {
                for(I idx = tid; idx < 3 * nb; idx += BS)
                {
                    const I m = mtop + idx / 3;
                    const I r = 1 + idx % 3;
                    vv(r, m) = vb[3 * (m - 1) + r - 1];
                }
                __syncthreads();
            }
        }

        // apply the reflection of bulge m (2-by-2 if m == m22) from the right to the
        // pair or triplet of columns starting at column c of the matrix A
        auto apply_right = [&](const I m, T& a1, T& a2, T& a3) {
            const T v1 = vv(1, m);
            const T v2 = vv(2, m);
            if(m == m22 && bmp22)
            {
                T refsum = v1 * (a1 + v2 * a2);
                a1 = a1 - refsum;
                a2 = a2 - refsum * conj(v2);
            }
            else
            {
                const T v3 = vv(3, m);
                T refsum = v1 * (a1 + v2 * a2 + v3 * a3);
                a1 = a1 - refsum;
                a2 = a2 - refsum * conj(v2);
                a3 = a3 - refsum * conj(v3);
            }
        };
        const I jtop = accum ? std::max(ktop, incol) : (wantt ? I(1) : ktop);

        // 2. multiply H by reflections from the left. The bulges act on disjoint rows, so
        //    all the (bulge, column) pairs are independent; they are distributed among the
        //    threads with the bulge index running fastest (the rows k+1:k+3 of consecutive
        //    bulges are contiguous in memory). Bulge m acts on the columns j >= k+1.
        const I jbot = accum ? std::min(ndcol, kbot) : (wantt ? n : kbot);
        {
            const I j0 = std::max(ktop, krcol);
            const I ncols = jbot - j0 + 1;
            // apply the reflection of bulge m from the left to column j
            auto apply_left = [&](const I m, const I j) {
                const I k = krcol + 3 * (m - 1);
                if(bmp22 && m == m22)
                {
                    if(j >= std::max(k + 1, ktop))
                    {
                        T refsum = conj(vv(1, m22)) * (h(k + 1, j) + conj(vv(2, m22)) * h(k + 2, j));
                        h(k + 1, j) = h(k + 1, j) - refsum;
                        h(k + 2, j) = h(k + 2, j) - refsum * vv(2, m22);
                    }
                }
                else if(j >= k + 1)
                {
                    T refsum = conj(vv(1, m))
                        * (h(k + 1, j) + conj(vv(2, m)) * h(k + 2, j) + conj(vv(3, m)) * h(k + 3, j));
                    h(k + 1, j) = h(k + 1, j) - refsum;
                    h(k + 2, j) = h(k + 2, j) - refsum * vv(2, m);
                    h(k + 3, j) = h(k + 3, j) - refsum * vv(3, m);
                }
            };

            // (with G > 1, thread-block 0 updates the columns k+1:k+6 of each bulge, and
            // the others the columns to their right, together with most of the rows above
            // each bulge from the right; see laqr5_chunk_block)
            if(nb > 0 && ncols > 0 && G == 1)
            {
                for(I idx = tid; idx < nb * ncols; idx += BS)
                    apply_left(mtop + idx % nb, j0 + idx / nb);
            }
            else if(nb > 0 && ncols > 0 && lead)
            {
                for(I idx = tid; idx < nb * 6; idx += BS)
                {
                    const I m = mtop + idx % nb;
                    const I j = krcol + 3 * (m - 1) + 1 + idx / nb;
                    if(j >= j0 && j <= jbot)
                        apply_left(m, j);
                }
            }
            else if(nb > 0 && G > 1 && !lead)
            {
                // work items (distributed together, so that each thread has few):
                // (a) the 3-by-3 blocks of the bulges m1 <= m-2 (rows k1+1:k1+3) and m
                //     (columns k+1:k+3), from the left by bulge m1, then from the right by
                //     bulge m (nb*nb items, the others skipped);
                // (b) the rows jtop:k(mtop) above the first bulge, from the right only;
                // (c) the columns to the right of the last bulge, from the left only
                const I kf = krcol + 3 * (mtop - 1);
                const I nfree = std::max(std::min(kbot, kf) - jtop + 1, I(0));
                const I jt0 = std::max(krcol + 3 * (mlast - 1) + 4, j0);
                const I nt = std::max(jbot - jt0 + 1, I(0));
                const I na = nb * nb;
                const I nf = nb * nfree;
                for(I idx = (w - 1) * BS + tid; idx < na + nf + nb * nt; idx += (G - 1) * BS)
                {
                    if(idx < na)
                    {
                        const I m1 = mtop + idx % nb;
                        const I m = mtop + idx / nb;
                        if(m1 > m - 2)
                            continue;
                        const I k1 = krcol + 3 * (m1 - 1);
                        const I k = krcol + 3 * (m - 1);
                        // (the 2-by-2 bulge m22 has two columns; its third, kbot+1, is not
                        // touched. The loops have constant bounds, so that a stays in
                        // registers)
                        const bool c3 = !(m == m22 && bmp22);
                        T a[3][3] = {};
                        for(int r = 0; r < 3; r++)
                            for(int c = 0; c < 3; c++)
                                if(c < 2 || c3)
                                    a[r][c] = h(k1 + 1 + r, k + 1 + c);
                        const T u1 = conj(vv(1, m1));
                        const T u2 = vv(2, m1);
                        const T u3 = vv(3, m1);
                        for(int c = 0; c < 3; c++)
                        {
                            if(k + 1 + c < j0 || k + 1 + c > jbot)
                                continue;
                            T refsum = u1 * (a[0][c] + conj(u2) * a[1][c] + conj(u3) * a[2][c]);
                            a[0][c] = a[0][c] - refsum;
                            a[1][c] = a[1][c] - refsum * u2;
                            a[2][c] = a[2][c] - refsum * u3;
                        }
                        if(vv(1, m) != T(0))
                            for(int r = 0; r < 3; r++)
                                if(k1 + 1 + r >= jtop && k1 + 1 + r <= kbot)
                                    apply_right(m, a[r][0], a[r][1], a[r][2]);
                        for(int r = 0; r < 3; r++)
                            for(int c = 0; c < 3; c++)
                                if(c < 2 || c3)
                                    h(k1 + 1 + r, k + 1 + c) = a[r][c];
                    }
                    else if(idx < na + nf)
                    {
                        const I m = mtop + (idx - na) / nfree;
                        const I r = jtop + (idx - na) % nfree;
                        const I k = krcol + 3 * (m - 1);
                        if(vv(1, m) == T(0) || r > k - 1)
                            continue;
                        T a3dummy = T(0);
                        T& a3 = (m == m22 && bmp22) ? a3dummy : h(r, k + 3);
                        apply_right(m, h(r, k + 1), h(r, k + 2), a3);
                    }
                    else
                    {
                        const I m = mtop + (idx - na - nf) % nb;
                        const I j = jt0 + (idx - na - nf) / nb;
                        if(j >= krcol + 3 * (m - 1) + 7)
                            apply_left(m, j);
                    }
                }
            }
        }

        // (with G > 1, the multiplications from the right of each thread-block only use
        // entries that the same thread-block multiplied from the left)
        __syncthreads();

        // 3. multiply H by reflections from the right (and, without accum, Z; with accum
        //    they were stored in Vbuf). Delay filling in the last row until the vigilant deflation check
        //    is complete. The bulges act on disjoint columns, so all the (bulge, row) pairs
        //    are independent and are distributed among the threads.
        {
            // H: rows jtop:min(kbot,k+3) of each bulge (the range of the last bulge is the
            // longest; shorter ranges skip the extra rows). With G > 1, thread-block 0
            // updates rows max(jtop,k-2):min(kbot,k+3) (k:k+3 for the first bulge), and the
            // others the rows above them, in step 2.
            if(nb > 0 && G > 1 && lead)
            {
                for(I idx = tid; idx < nb * 6; idx += BS)
                {
                    const I m = mtop + idx / 6;
                    const I k = krcol + 3 * (m - 1);
                    const I j = k - 2 + idx % 6;
                    if(vv(1, m) == T(0) || j < jtop || j > std::min(kbot, k + 3)
                       || (m == mtop && j < k))
                        continue;
                    T a3dummy = T(0);
                    T& a3 = (m == m22 && bmp22) ? a3dummy : h(j, k + 3);
                    apply_right(m, h(j, k + 1), h(j, k + 2), a3);
                }
            }
            else if(nb > 0 && G == 1)
            {
                const I kmax = krcol + 3 * (mlast - 1);
                const I nrows = std::min(kbot, kmax + 3) - jtop + 1;
                for(I idx = tid; idx < nb * nrows; idx += BS)
                {
                    const I m = mtop + idx / nrows;
                    const I j = jtop + idx % nrows;
                    const I k = krcol + 3 * (m - 1);
                    if(vv(1, m) == T(0) || j > std::min(kbot, k + 3))
                        continue;
                    T a3dummy = T(0);
                    T& a3 = (m == m22 && bmp22) ? a3dummy : h(j, k + 3);
                    apply_right(m, h(j, k + 1), h(j, k + 2), a3);
                }
            }

            if(!accum && wantz && nb > 0)
            {
                // U is not accumulated, so update Z now by multiplying by reflections from
                // the right
                const I nrows = ihiz - iloz + 1;
                for(I idx = tid; idx < nb * nrows; idx += BS)
                {
                    const I m = mtop + idx / nrows;
                    const I j = iloz + idx % nrows;
                    if(vv(1, m) == T(0))
                        continue;
                    const I k = krcol + 3 * (m - 1);
                    T a3dummy = T(0);
                    T& a3 = (m == m22 && bmp22) ? a3dummy : z(j, k + 3);
                    apply_right(m, z(j, k + 1), z(j, k + 2), a3);
                }
            }
        }
        __syncthreads();

        // 4. vigilant deflation check (one thread per bulge; the checks of bulges three rows
        //    apart read and write disjoint entries). At the bottom (krcol = kbot-2) the
        //    checks at k = kbot-2 and k = kbot-1 are one row apart: each may read the
        //    subdiagonal entry that the other one sets to zero, so all the checks of this
        //    step run in order on one thread, as in LAPACK.
        {
            I mstart = mtop;
            if(krcol + 3 * (mstart - 1) < ktop)
                mstart = mstart + 1;
            I mend = mbot;
            if(bmp22)
                mend = mend + 1;
            if(krcol == kbot - 2)
            {
                mend = mend + 1;
                if(lead && tid == 0)
                    for(I m = mstart; m <= mend; m++)
                        vigilant(std::min(kbot - 1, krcol + 3 * (m - 1)));
            }
            else
            {
                for(I m = mstart + tid; lead && m <= mend; m += BS)
                    vigilant(std::min(kbot - 1, krcol + 3 * (m - 1)));
            }
        }
        __syncthreads();

        // 5. fill in the last row of each bulge
        {
            const I mend = std::min(nbmps, (kbot - krcol - 1) / 3);
            for(I m = mtop + tid; lead && m <= mend; m += BS)
            {
                const I k = krcol + 3 * (m - 1);
                T refsum = vv(1, m) * vv(3, m) * h(k + 4, k + 3);
                h(k + 4, k + 1) = -refsum;
                h(k + 4, k + 2) = -refsum * conj(vv(2, m));
                h(k + 4, k + 3) = h(k + 4, k + 3) - refsum * conj(vv(3, m));
            }
        }
        __syncthreads();
    }
}

/** LAQR5_BUILD_U_BLOCK forms rows r0:r0+nr-1 of the kdu-by-kdu matrix U of one chunk
    of the sweep (laqr5_chunk_block with accum), the product of the reflections stored in
    Vbuf, with the rows in shared memory (tile, nr-by-kdu with leading dimension nr).
    Within a step the reflections act on disjoint columns; one barrier per step. **/
template <int BS, typename T, typename I>
__device__ void laqr5_build_u_block(const I ktop,
                                    const I kbot,
                                    const I nbmps,
                                    const I incol,
                                    const T* Vbuf,
                                    const I r0,
                                    const I nr,
                                    T* tile,
                                    T* U,
                                    const I ldu)
{
    const I tid = hipThreadIdx_x;
    const I kdu = 6 * nbmps - 3;
    const I ldvb = 3 * (nbmps + 1);
    const I j0 = std::max(I(1), ktop - incol);
    auto t = [&](const I i, const I j) -> T& { return tile[(i - r0) + (j - 1) * nr]; };

    for(I e = tid; e < nr * kdu; e += BS)
    {
        const I i = r0 + e % nr;
        const I j = 1 + e / nr;
        t(i, j) = (i == j) ? T(1) : T(0);
    }
    __syncthreads();

    const I krlast = std::min(incol + 3 * nbmps - 3, kbot - 2);
    for(I krcol = incol; krcol <= krlast; krcol++)
    {
        const I mtop = std::max(I(1), ((ktop - 1) - krcol + 2) / 3 + 1);
        const I mbot = std::min(nbmps, (kbot - krcol) / 3);
        const I m22 = mbot + 1;
        const bool bmp22 = (mbot < nbmps) && (krcol + 3 * (m22 - 1) == kbot - 2);
        const I mlast = mbot + (bmp22 ? 1 : 0);
        const I nb = mlast - mtop + 1;
        const T* vb = Vbuf + (krcol - incol) * ldvb;
        for(I idx = tid; idx < nb * nr; idx += BS)
        {
            const I i = r0 + idx % nr;
            const I m = mtop + idx / nr;
            if(i < j0)
                continue;
            const T v1 = vb[3 * (m - 1)];
            if(v1 == T(0))
                continue;
            const T v2 = vb[3 * (m - 1) + 1];
            const I kms = krcol + 3 * (m - 1) - incol;
            if(m == m22 && bmp22)
            {
                T a1 = t(i, kms + 1), a2 = t(i, kms + 2);
                T refsum = v1 * (a1 + v2 * a2);
                t(i, kms + 1) = a1 - refsum;
                t(i, kms + 2) = a2 - refsum * conj(v2);
            }
            else
            {
                const T v3 = vb[3 * (m - 1) + 2];
                T a1 = t(i, kms + 1), a2 = t(i, kms + 2), a3 = t(i, kms + 3);
                T refsum = v1 * (a1 + v2 * a2 + v3 * a3);
                t(i, kms + 1) = a1 - refsum;
                t(i, kms + 2) = a2 - refsum * conj(v2);
                t(i, kms + 3) = a3 - refsum * conj(v3);
            }
        }
        __syncthreads();
    }

    for(I e = tid; e < nr * kdu; e += BS)
    {
        const I i = r0 + e % nr;
        const I j = 1 + e / nr;
        U[idx2D(i - 1, j - 1, ldu)] = t(i, j);
    }
}

/** LAQR5_LEFT_APPLY_BLOCK applies the reflections of one chunk of the sweep, stored in
    Vbuf by laqr5_chunk_block (with accum), from the left to column j > ndcol of H, as
    the chase does to the columns of its window (rows incol+1:ndcol; phase 2 of each
    step). The column is kept in shared memory (col, kdu entries); the bulges of a step
    act on disjoint rows. This updates the columns that the next chunk will use, so that
    the matrix-matrix products with U can run concurrently with the next chunk. **/
template <int BS, typename T, typename I>
__device__ void laqr5_left_apply_block(const I n,
                                       const I ktop,
                                       const I kbot,
                                       const I nbmps,
                                       const I incol,
                                       const T* Vbuf,
                                       T* H,
                                       const I ldh,
                                       const I j,
                                       T* col)
{
    const I tid = hipThreadIdx_x;
    const I kdu = 6 * nbmps - 3;
    const I ldvb = 3 * (nbmps + 1);
    // (the reflections act on rows ktop:kbot; the rows below kbot may be the workspace of
    // the products with U of the same chunk, which run concurrently on the side stream, so
    // they must not be written, even with unchanged values)
    const I r0 = std::max(incol + 1, ktop);
    const I r1 = std::min(incol + kdu, kbot);
    auto c = [&](const I r) -> T& { return col[r - incol - 1]; };

    for(I r = r0 + tid; r <= r1; r += BS)
        c(r) = H[idx2D(r - 1, j - 1, ldh)];
    __syncthreads();

    const I krlast = std::min(incol + 3 * nbmps - 3, kbot - 2);
    for(I krcol = incol; krcol <= krlast; krcol++)
    {
        const I mtop = std::max(I(1), ((ktop - 1) - krcol + 2) / 3 + 1);
        const I mbot = std::min(nbmps, (kbot - krcol) / 3);
        const I m22 = mbot + 1;
        const bool bmp22 = (mbot < nbmps) && (krcol + 3 * (m22 - 1) == kbot - 2);
        const I mlast = mbot + (bmp22 ? 1 : 0);
        const T* vb = Vbuf + (krcol - incol) * ldvb;
        for(I m = mtop + tid; m <= mlast; m += BS)
        {
            const I k = krcol + 3 * (m - 1);
            const T v1 = vb[3 * (m - 1)];
            const T v2 = vb[3 * (m - 1) + 1];
            if(bmp22 && m == m22)
            {
                if(j >= std::max(k + 1, ktop))
                {
                    T refsum = conj(v1) * (c(k + 1) + conj(v2) * c(k + 2));
                    c(k + 1) = c(k + 1) - refsum;
                    c(k + 2) = c(k + 2) - refsum * v2;
                }
            }
            else if(j >= k + 1)
            {
                const T v3 = vb[3 * (m - 1) + 2];
                T refsum = conj(v1) * (c(k + 1) + conj(v2) * c(k + 2) + conj(v3) * c(k + 3));
                c(k + 1) = c(k + 1) - refsum;
                c(k + 2) = c(k + 2) - refsum * v2;
                c(k + 3) = c(k + 3) - refsum * v3;
            }
        }
        __syncthreads();
    }

    for(I r = r0 + tid; r <= r1; r += BS)
        H[idx2D(r - 1, j - 1, ldh)] = c(r);
}

/** Indices of the status array of the multishift QR iteration. **/
enum laqr0_status_index
{
    LAQR0_KTOP = 0,
    LAQR0_NW,
    LAQR0_NDEC,
    LAQR0_LS,
    LAQR0_LD,
    LAQR0_JW,
    LAQR0_KWTOP,
    LAQR0_UPDATE,
    LAQR0_SWEEP,
    LAQR0_NS,
    LAQR0_KS,
    LAQR0_KBOT,
    LAQR0_CORE,
    LAQR0_TOOFEW,
    LAQR0_DEFER,
    LAQR0_STATUS_SIZE
};

/** Indices of the (complex) scalar status array of the multishift QR iteration. **/
enum laqr0_status_scalar_index
{
    LAQR0_SPIKE_IN = 0,
    LAQR0_SPIKE_OUT,
    LAQR0_STATUS_SCALAR_SIZE
};

/** LAQR0_SHIFTS_TAIL finishes the shift selection of ZLAQR0 once the candidate
    shifts W(ks:kbot2) are known (W(i) is Wb[i - wb]): if fail (ZLAHQR could not
    compute the eigenvalues of the trailing submatrix), it uses the eigenvalues of the
    trailing 2-by-2 block hb = H(kbot2-1:kbot2, kbot2-1:kbot2) (column-major; only
    hb[3] = H(kbot2,kbot2) is needed otherwise), if sort it sorts the shifts, and then
    it chooses the ns shifts to use, W(ks:kbot2) on exit. On the device it is
    executed by all the threads of a thread-block of BS threads; on the host, with
    BS = 1. **/
template <int BS, typename T, typename I>
__host__ __device__ void laqr0_shifts_tail(const bool sort,
                                           const bool fail,
                                           I& ns,
                                           I& ks,
                                           const I kbot2,
                                           T* Wb,
                                           const I wb,
                                           const T* hb)
{
    using S = decltype(std::real(T{}));

    const I tid = hqr_tid();
    auto w = [&](const I i) -> T& { return Wb[i - wb]; };

    if(fail)
    {
        // in case of a rare QR failure use eigenvalues of the trailing 2-by-2
        // principal submatrix; scale to avoid overflows, underflows and subnormals
        S sc = hqr_cabs1(hb[0]) + hqr_cabs1(hb[1]) + hqr_cabs1(hb[2]) + hqr_cabs1(hb[3]);
        T aa = hb[0] / sc;
        T cc = hb[1] / sc;
        T bb = hb[2] / sc;
        T dd = hb[3] / sc;
        T tr2 = (aa + dd) / S(2);
        T det = (aa - tr2) * (dd - tr2) - bb * cc;
        T rtdisc = hqr_csqrt(T(-det.real(), -det.imag()));
        hqr_sync();
        if(tid == 0)
        {
            w(kbot2 - 1) = sc * (tr2 + rtdisc);
            w(kbot2) = sc * (tr2 - rtdisc);
        }
        ks = kbot2 - 1;
    }

    if(sort && kbot2 - ks + 1 > ns)
    {
        // sort the shifts (helps a little): bubble sort into decreasing |Re|+|Im|
        hqr_sync();
        if(tid == 0)
        {
            bool sorted = false;
            for(I k = kbot2; k >= ks + 1 && !sorted; k--)
            {
                sorted = true;
                for(I i = ks; i <= k - 1; i++)
                {
                    if(hqr_cabs1(w(i)) < hqr_cabs1(w(i + 1)))
                    {
                        sorted = false;
                        T swp = w(i);
                        w(i) = w(i + 1);
                        w(i + 1) = swp;
                    }
                }
            }
        }
    }
    hqr_sync();

    // if there are only two shifts, then use only one
    if(kbot2 - ks + 1 == 2)
    {
        T w1 = w(kbot2 - 1);
        T w2 = w(kbot2);
        T hkk = hb[3];
        hqr_sync();
        if(tid == 0)
        {
            if(hqr_cabs1(w2 - hkk) < hqr_cabs1(w1 - hkk))
                w(kbot2 - 1) = w2;
            else
                w(kbot2) = w1;
        }
    }

    // use up to ns of the smallest magnitude shifts; if there are not ns shifts
    // available, then use them all, possibly dropping one to make the number of
    // shifts even
    ns = std::min(ns, kbot2 - ks + 1);
    ns = ns - ns % 2;
    ks = kbot2 - ns + 1;
}

/*
 * ===========================================================================
 *    One iteration of the main loop of LAPACK ZLAQR0 (version 3.9.0, whose
 *    workspace bounds, e.g. NSMAX = (n+6)/9, match the sweep of laqr5_chunk_block;
 *    3.12 uses NSMAX = (n-3)/6), except for the off-window updates of the AED, the
 *    extra shifts when the AED provides too few (laqr0_toofew_block) and the sweep
 *    itself, which are left to the caller. LAPACK computes the Schur forms of the
 *    deflation window and of the too-few-shifts submatrix with ZLAQR4 for sizes above
 *    NMIN = 75, and with ZLAHQR otherwise. Here, on the device, windows larger than
 *    LAQR4_NMIN (64, the largest size of the shared memory kernel lahqr_lds_block) use
 *    laqr4_block, deferred to laqr0_core4_block, and the other windows, the
 *    too-few-shifts submatrix and all the windows of the hybrid mode use ZLAHQR (the
 *    window is capped by hseqr_aed_window_cap, but can still grow to (n-1)/3 when the
 *    iterations stall).
 *    It is split in two parts around the core of the aggressive early deflation
 *    (aed_core_block), which runs on the device (laqr0_iteration_block) or on the
 *    host (hybrid mode):
 *    - laqr0_part1_block: locate the active block (ktop: the last exact zero on the
 *      subdiagonal), select the deflation window size (nw, ndec), and set up the
 *      AED, using the workspace below the subdiagonal of H as in LAPACK
 *      (V = H(kv,1), T = H(kv,kt), work = H(kwv,1));
 *    - laqr0_part2_block: finish the AED, decide whether a sweep is needed and, if so,
 *      select the shifts (the AED shifts or exceptional shifts), leaving them in
 *      W(ks:kbot); if the AED provides too few, it sets status[LAQR0_TOOFEW] and the
 *      caller completes the selection with laqr0_toofew_block after the off-window
 *      updates.
 *    The caller provides kbot, ndfl, and the previous nw and ndec; nwr, nwmax, nsr
 *    and nsmax are the tuning parameters of ZLAQR0 (with an unlimited LWORK). The
 *    results are left in status (see laqr0_status_index) and statusT.
 * ===========================================================================
 */
template <int BS, typename T, typename I>
__device__ void laqr0_part1_block(const I n,
                                  const I ilo,
                                  const I kbot,
                                  const I ndfl,
                                  const I nw_prev,
                                  const I ndec_prev,
                                  const I nwr,
                                  const I nwmax,
                                  T* H,
                                  const I ldh,
                                  T* W,
                                  I* status,
                                  T* statusT,
                                  I (*s_ired)[HQR_RED(BS)],
                                  int& ibuf)
{
    const I tid = hqr_tid();
    auto h = [&](const I i, const I j) -> T& { return H[idx2D(i - 1, j - 1, ldh)]; };

    constexpr I kexnw = 5;

    // locate active block
    I kfound = 0;
    for(I k = kbot - tid; k >= ilo + 1; k -= BS)
    {
        if(h(k, k - 1) == T(0))
        {
            kfound = k;
            break;
        }
    }
    kfound = hqr_block_max<BS>(kfound, s_ired, ibuf);
    const I ktop = (kfound > 0) ? kfound : ilo;

    // select deflation window size
    const I nh = kbot - ktop + 1;
    const I nwupbd = std::min(nh, nwmax);
    I nw = (ndfl < kexnw) ? std::min(nwupbd, nwr) : std::min(nwupbd, 2 * nw_prev);
    if(nw < nwmax)
    {
        if(nw >= nh - 1)
            nw = nh;
        else
        {
            const I kwtop = kbot - nw + 1;
            if(hqr_cabs1(h(kwtop, kwtop - 1)) > hqr_cabs1(h(kwtop - 1, kwtop - 2)))
                nw = nw + 1;
        }
    }
    I ndec = ndec_prev;
    if(ndfl < kexnw)
        ndec = -1;
    else if(ndec >= 0 || nw >= nwupbd)
    {
        ndec = ndec + 1;
        if(nw - ndec < 2)
            ndec = 0;
        nw = nw - ndec;
    }

    // set up the aggressive early deflation, with the workspace below the subdiagonal
    const I kv = n - nw + 1;
    const I kt = nw + 1;
    I jw, kwtop, ns, nd;
    T s;
    const bool core = aed_setup_block<BS>(n, ktop, kbot, nw, H, ldh, W, &h(kv, kt), ldh, &h(kv, 1),
                                          ldh, jw, kwtop, s, ns, nd);

    if(tid == 0)
    {
        status[LAQR0_KTOP] = ktop;
        status[LAQR0_NW] = nw;
        status[LAQR0_NDEC] = ndec;
        status[LAQR0_JW] = jw;
        status[LAQR0_KWTOP] = kwtop;
        status[LAQR0_CORE] = core ? 1 : 0;
        status[LAQR0_LS] = ns;
        status[LAQR0_LD] = nd;
        status[LAQR0_UPDATE] = 0;
        status[LAQR0_DEFER] = 0;
        statusT[LAQR0_SPIKE_IN] = s;
        statusT[LAQR0_SPIKE_OUT] = T(0);
    }
    __syncthreads();
}

template <int BS, typename T, typename I>
__device__ void laqr0_part2_block(const I n,
                                  const I kbot,
                                  const I ndfl,
                                  const I nwmax,
                                  const I nsr,
                                  const I nsmax,
                                  T* H,
                                  const I ldh,
                                  T* W,
                                  I* status,
                                  T* statusT,
                                  I (*s_ired)[HQR_RED(BS)],
                                  int& ibuf)
{
    using S = decltype(std::real(T{}));

    const I tid = hqr_tid();
    auto h = [&](const I i, const I j) -> T& { return H[idx2D(i - 1, j - 1, ldh)]; };

    constexpr I kexsh = 6;
    constexpr I nmin = 75; // LAPACK NMIN (IPARMQ, ISPEC = 12)
    constexpr I nibble = 14;
    const S wilk1 = S(0.75);

    const I ktop = status[LAQR0_KTOP];
    const I nw = status[LAQR0_NW];
    const I jw = status[LAQR0_JW];
    const I kwtop = status[LAQR0_KWTOP];
    const I ls = status[LAQR0_LS];
    const I ld = status[LAQR0_LD];
    const bool update = status[LAQR0_UPDATE] != 0;
    const T spike = statusT[LAQR0_SPIKE_OUT];
    const I kt = nw + 1;
    const I kv = n - nw + 1;
    __syncthreads();

    // finish the aggressive early deflation
    aed_finish_block<BS>(kwtop, jw, H, ldh, &h(kv, kt), ldh, update, spike);

    // adjust kbot accounting for new deflations; ks points to the shifts
    const I kbot2 = kbot - ld;
    I ks = kbot2 - ls + 1;

    // skip an expensive QR sweep if there is a (partly heuristic) reason to expect that
    // many eigenvalues will deflate without it
    const bool sweep
        = (ld == 0) || ((100 * ld <= nw * nibble) && (kbot2 - ktop + 1 > std::min(nmin, nwmax)));
    I ns = 0;
    if(sweep)
    {
        // nominal number of simultaneous shifts
        ns = std::min(nsmax, std::min(nsr, std::max(I(2), kbot2 - ktop)));
        ns = ns - ns % 2;

        const bool exceptional = (ndfl % kexsh == 0);
        if(exceptional)
        {
            // exceptional shifts
            ks = kbot2 - ns + 1;
            __syncthreads();
            for(I i = kbot2 - 2 * tid; i >= ks + 1; i -= 2 * BS)
            {
                T wi = T(h(i, i).real() + wilk1 * hqr_cabs1(h(i, i - 1)), h(i, i).imag());
                W[i - 1] = wi;
                W[i - 2] = wi;
            }
        }
        else if(kbot2 - ks + 1 <= ns / 2)
        {
            // got ns/2 or fewer shifts? They are computed later, by laqr0_toofew_block,
            // from the trailing ns-by-ns submatrix H(ks:kbot2, ks:kbot2), once the
            // off-window updates of the AED (which change its rows above the window) are
            // done, as in LAPACK
            ks = kbot2 - ns + 1;
            __syncthreads();
            if(tid == 0)
            {
                status[LAQR0_SWEEP] = 1;
                status[LAQR0_NS] = ns;
                status[LAQR0_KS] = ks;
                status[LAQR0_KBOT] = kbot2;
                status[LAQR0_TOOFEW] = 1;
            }
            return;
        }

        // trailing 2-by-2 block of the active part
        T hb[4];
        hb[3] = h(kbot2, kbot2);
        if(kbot2 >= 2)
        {
            hb[0] = h(kbot2 - 1, kbot2 - 1);
            hb[1] = h(kbot2, kbot2 - 1);
            hb[2] = h(kbot2 - 1, kbot2);
        }
        laqr0_shifts_tail<BS>(!exceptional, false, ns, ks, kbot2, W, I(1), hb);
    }
    __syncthreads();

    if(tid == 0)
    {
        status[LAQR0_TOOFEW] = 0;
        status[LAQR0_SWEEP] = sweep ? 1 : 0;
        status[LAQR0_NS] = ns;
        status[LAQR0_KS] = ks;
        status[LAQR0_KBOT] = kbot2;
    }
}

/** LAQR0_TOOFEW_BLOCK completes the shift selection when the AED provided too few
    shifts (status[LAQR0_TOOFEW] set by laqr0_part2_block): the shifts are the
    eigenvalues of the trailing submatrix H(ks:kbot2, ks:kbot2) (ns-by-ns), computed
    with ZLAHQR in the workspace H(n-ns+1:n, 1:ns) below the subdiagonal. It must run
    after the off-window updates of the AED.
    With the default settings (ns capped by the deflation window), this path is only
    reached when the window has shrunk after several iterations without deflations, or
    when the Schur form of the window fails. **/
template <int BS, typename T, typename I>
__device__ void
    laqr0_toofew_block(const I n, T* H, const I ldh, T* W, I* status, I (*s_ired)[HQR_RED(BS)], int& ibuf)
{
    const I tid = hqr_tid();
    auto h = [&](const I i, const I j) -> T& { return H[idx2D(i - 1, j - 1, ldh)]; };

    I ns = status[LAQR0_NS];
    I ks = status[LAQR0_KS];
    const I kbot2 = status[LAQR0_KBOT];
    const I kt = n - ns + 1;
    __syncthreads();

    // trailing 2-by-2 block of the active part
    T hb[4];
    hb[0] = h(kbot2 - 1, kbot2 - 1);
    hb[1] = h(kbot2, kbot2 - 1);
    hb[2] = h(kbot2 - 1, kbot2);
    hb[3] = h(kbot2, kbot2);

    for(I j = 0; j < ns; j++)
        for(I i = tid; i < ns; i += BS)
            h(kt + i, 1 + j) = h(ks + i, ks + j);
    __syncthreads();
    const I inf = lahqr_block<BS>(false, false, ns, I(1), ns, &h(kt, 1), ldh, W + (ks - 1), I(1),
                                  I(1), (T*)nullptr, I(1), s_ired, ibuf);
    __syncthreads();
    ks = ks + inf;

    laqr0_shifts_tail<BS>(true, ks >= kbot2, ns, ks, kbot2, W, I(1), hb);
    __syncthreads();

    if(tid == 0)
    {
        status[LAQR0_NS] = ns;
        status[LAQR0_KS] = ks;
        status[LAQR0_TOOFEW] = 0;
    }
}

/** HQR_BLOCK_GEMM_COPY computes C (m-by-nn) = op(A) * B (op(A) = A, or A^H if conjA; k
    is the inner dimension) in the scratch Wk (ldw), then copies it back into C (which
    may be A or B). All the threads of the block must call it. **/
template <int BS, typename T, typename I>
__device__ void hqr_block_gemm_copy(const bool conjA,
                                    const I m,
                                    const I nn,
                                    const I k,
                                    const T* A,
                                    const I lda,
                                    const T* B,
                                    const I ldb,
                                    T* C,
                                    const I ldc,
                                    T* Wk,
                                    const I ldw)
{
    const I tid = hqr_tid();
    if(m <= 0 || nn <= 0)
        return;
    for(I idx = tid; idx < m * nn; idx += BS)
    {
        const I i = idx % m;
        const I j = idx / m;
        T sum = T(0);
        for(I l = 0; l < k; l++)
            sum += (conjA ? conj(A[idx2D(l, i, lda)]) : A[idx2D(i, l, lda)]) * B[idx2D(l, j, ldb)];
        Wk[idx2D(i, j, ldw)] = sum;
    }
    hqr_sync();
    for(I idx = tid; idx < m * nn; idx += BS)
    {
        const I i = idx % m;
        const I j = idx / m;
        C[idx2D(i, j, ldc)] = Wk[idx2D(i, j, ldw)];
    }
    hqr_sync();
}

/** LAQR4_BLOCK computes the Schur form of the n-by-n upper Hessenberg matrix H and
    multiplies Z (n-by-n) by the Schur vectors, with the multishift QR algorithm and
    aggressive early deflation of LAPACK ZLAQR4 (ZLAQR0 without recursion: its deflation
    windows use ZLAHQR), for the deflation windows of aed_core_block. The iterations are
    those of hseqr_multishift, with wantt and wantz, and the off-window updates and the
    sweeps are applied directly (the reflections are not accumulated). The part of H
    below its first subdiagonal is the workspace, as in LAPACK. st and stT are the
    status arrays (see laqr0_status_index) and sV holds LAQR4_MAX_SHIFTS/2+2
    reflections (in shared memory on the device). The eigenvalues are returned in W.
    Returns 0, or kbot > 0 if the iterations failed to converge (as in LAPACK). All the
    threads of the block must call it. **/
template <int BS, typename T, typename I>
__device__ I laqr4_block(const I n,
                         T* H,
                         const I ldh,
                         T* W,
                         T* Z,
                         const I ldz,
                         I* st,
                         T* stT,
                         T (*sV)[3],
                         I (*s_ired)[HQR_RED(BS)],
                         decltype(std::real(T{})) (*s_sred)[HQR_RED(BS)],
                         int& ibuf,
                         int& sbuf,
                         T* lds_ws)
{
    const I tid = hqr_tid();
    auto h = [&](const I i, const I j) -> T& { return H[idx2D(i - 1, j - 1, ldh)]; };

    // tuning parameters (LAPACK IPARMQ for the order n, and ZLAQR4 with an unlimited LWORK)
    I ns_ip, nw_ip;
    hseqr_iparmq(n, ns_ip, nw_ip);
    const I nwr = std::min(n, std::min((n - 1) / 3, std::max(I(2), nw_ip)));
    I nsr = std::min(ns_ip, std::min((n + 6) / 9, n - 1));
    nsr = std::min(nsr, I(LAQR4_MAX_SHIFTS));
    nsr = std::max(I(2), nsr - nsr % 2);
    const I nwmax = (n - 1) / 3;
    I nsmax = (n + 6) / 9;
    nsmax = std::min(nsmax - nsmax % 2, I(LAQR4_MAX_SHIFTS));
    const I itmax = 30 * std::max(I(10), n);

    I kbot = n;
    I nw = nwmax;
    I ndec = -1;
    I ndfl = 1;
    for(I it = 1; it <= itmax; it++)
    {
        if(kbot < 1)
            return 0;

        // active block, deflation window, AED (with ZLAHQR for the window) and shifts
        laqr0_part1_block<BS>(n, I(1), kbot, ndfl, nw, ndec, nwr, nwmax, H, ldh, W, st, stT, s_ired,
                              ibuf);
        hqr_sync();
        if(st[LAQR0_CORE])
        {
            const I nwc = st[LAQR0_NW];
            const I jw = st[LAQR0_JW];
            const I kwtop = st[LAQR0_KWTOP];
            const T s = stT[LAQR0_SPIKE_IN];
            const I kv = n - nwc + 1;
            const I kt = nwc + 1;
            const I kwv = nwc + 2;
            I ls, ld;
            bool update;
            T spike;
            hqr_sync();
            aed_core_block<BS, T, I, false>(n, jw, s, &h(kv, kt), ldh, &h(kv, 1), ldh, &h(kwv, 1),
                                            W + (kwtop - 1), ls, ld, update, spike, s_ired, s_sred,
                                            ibuf, sbuf, lds_ws);
            if(tid == 0)
            {
                st[LAQR0_LS] = ls;
                st[LAQR0_LD] = ld;
                st[LAQR0_UPDATE] = update ? 1 : 0;
                stT[LAQR0_SPIKE_OUT] = spike;
            }
        }
        hqr_sync();
        laqr0_part2_block<BS>(n, kbot, ndfl, nwmax, nsr, nsmax, H, ldh, W, st, stT, s_ired, ibuf);
        hqr_sync();

        const I ktop = st[LAQR0_KTOP];
        nw = st[LAQR0_NW];
        ndec = st[LAQR0_NDEC];
        const I ld = st[LAQR0_LD];

        // off-window updates of the AED, through the workspace below the subdiagonal:
        // V = H(kv,1), WV = H(kwv,1) (nve rows) and WH = H(kv,kt) (nho columns)
        if(st[LAQR0_UPDATE])
        {
            const I jw = st[LAQR0_JW];
            const I kwtop = st[LAQR0_KWTOP];
            const I kv = n - nw + 1;
            const I kt = nw + 1;
            const I nho = (n - nw - 1) - kt + 1;
            const I kwv = nw + 2;
            const I nve = (n - nw) - kwv + 1;
            T* V = &h(kv, 1);
            for(I krow = 1; krow <= kwtop - 1; krow += nve)
            {
                const I kln = std::min(nve, kwtop - krow);
                hqr_block_gemm_copy<BS>(false, kln, jw, jw, &h(krow, kwtop), ldh, V, ldh,
                                        &h(krow, kwtop), ldh, &h(kwv, 1), ldh);
            }
            for(I kcol = kbot + 1; kcol <= n; kcol += nho)
            {
                const I kln = std::min(nho, n - kcol + 1);
                hqr_block_gemm_copy<BS>(true, jw, kln, jw, V, ldh, &h(kwtop, kcol), ldh,
                                        &h(kwtop, kcol), ldh, &h(kv, kt), ldh);
            }
            for(I krow = 1; krow <= n; krow += nve)
            {
                const I kln = std::min(nve, n - krow + 1);
                hqr_block_gemm_copy<BS>(false, kln, jw, jw, &Z[idx2D(krow - 1, kwtop - 1, ldz)],
                                        ldz, V, ldh, &Z[idx2D(krow - 1, kwtop - 1, ldz)], ldz,
                                        &h(kwv, 1), ldh);
            }
        }

        // too few shifts from the AED
        if(st[LAQR0_TOOFEW])
        {
            laqr0_toofew_block<BS>(n, H, ldh, W, st, s_ired, ibuf);
            hqr_sync();
        }

        kbot = st[LAQR0_KBOT];

        // small-bulge multishift QR sweep, applied directly to H and Z
        const I ns = st[LAQR0_NS];
        if(st[LAQR0_SWEEP] && ns >= 2 && ktop < kbot)
        {
            const I ks = st[LAQR0_KS];
            const I nbmps = ns / 2;
            hqr_sync();
            if(tid == 0 && ktop + 2 <= kbot)
                h(ktop + 2, ktop) = T(0);
            hqr_sync();
            for(I incol = 3 * (1 - nbmps) + ktop - 1; incol <= kbot - 2; incol += 3 * nbmps - 2)
                laqr5_chunk_block<BS>(true, true, false, n, ktop, kbot, nbmps, incol, W + (ks - 1),
                                      H, ldh, I(1), n, Z, ldz, (T*)nullptr, sV);
        }

        // note progress (or the lack of it)
        ndfl = (ld > 0) ? 1 : ndfl + 1;
        hqr_sync();
    }

    // iteration limit exceeded
    return kbot;
}

/** LAQR0_ITERATION_BLOCK runs one iteration (both parts and the AED core) on the
    device. **/
template <int BS, typename T, typename I>
__device__ void laqr0_iteration_block(const I n,
                                      const I ilo,
                                      const I kbot,
                                      const I ndfl,
                                      const I nw_prev,
                                      const I ndec_prev,
                                      const I nwr,
                                      const I nwmax,
                                      const I nsr,
                                      const I nsmax,
                                      T* H,
                                      const I ldh,
                                      T* W,
                                      I* status,
                                      T* statusT)
{
    using S = decltype(std::real(T{}));
    __shared__ I s_ired[2][HQR_RED(BS)];
    __shared__ S s_sred[2][HQR_RED(BS)];
    // shared workspace of the Schur form of small deflation windows
    __shared__ T lds_ws[HQR_LDS_WS_SIZE];
    int ibuf = 0, sbuf = 0;

    const I tid = hqr_tid();
    auto h = [&](const I i, const I j) -> T& { return H[idx2D(i - 1, j - 1, ldh)]; };

    laqr0_part1_block<BS>(n, ilo, kbot, ndfl, nw_prev, ndec_prev, nwr, nwmax, H, ldh, W, status,
                          statusT, s_ired, ibuf);

    // large windows: the AED core and part 2 are left to laqr0_core4_block
    if(status[LAQR0_CORE] && status[LAQR0_JW] > LAQR4_NMIN)
    {
        hqr_sync();
        if(tid == 0)
            status[LAQR0_DEFER] = 1;
        return;
    }

    if(status[LAQR0_CORE])
    {
        const I nw = status[LAQR0_NW];
        const I jw = status[LAQR0_JW];
        const I kwtop = status[LAQR0_KWTOP];
        const T s = statusT[LAQR0_SPIKE_IN];
        const I kv = n - nw + 1;
        const I kt = nw + 1;
        const I kwv = nw + 2;
        I ns, nd;
        bool update;
        T spike;
        __syncthreads();
        aed_core_block<BS>(n, jw, s, &h(kv, kt), ldh, &h(kv, 1), ldh, &h(kwv, 1), W + (kwtop - 1),
                           ns, nd, update, spike, s_ired, s_sred, ibuf, sbuf, lds_ws);
        if(tid == 0)
        {
            status[LAQR0_LS] = ns;
            status[LAQR0_LD] = nd;
            status[LAQR0_UPDATE] = update ? 1 : 0;
            statusT[LAQR0_SPIKE_OUT] = spike;
        }
    }
    __syncthreads();

    laqr0_part2_block<BS>(n, kbot, ndfl, nwmax, nsr, nsmax, H, ldh, W, status, statusT, s_ired, ibuf);
}

/** LAQR0_CORE4_BLOCK completes an iteration that laqr0_iteration_block deferred
    (status[LAQR0_DEFER]) because of a large deflation window: the AED core, with the
    Schur form of the window computed by laqr4_block, and part 2. **/
template <int BS, typename T, typename I>
__device__ void laqr0_core4_block(const I n,
                                  const I kbot,
                                  const I ndfl,
                                  const I nwmax,
                                  const I nsr,
                                  const I nsmax,
                                  T* H,
                                  const I ldh,
                                  T* W,
                                  I* status,
                                  T* statusT)
{
    using S = decltype(std::real(T{}));
    __shared__ I s_ired[2][HQR_RED(BS)];
    __shared__ S s_sred[2][HQR_RED(BS)];
    __shared__ T lds_ws[HQR_LDS_WS_SIZE];
    __shared__ I st4[LAQR0_STATUS_SIZE];
    __shared__ T stT4[LAQR0_STATUS_SCALAR_SIZE];
    __shared__ T sV4[LAQR4_MAX_SHIFTS / 2 + 2][3];
    int ibuf = 0, sbuf = 0;

    const I tid = hqr_tid();
    auto h = [&](const I i, const I j) -> T& { return H[idx2D(i - 1, j - 1, ldh)]; };

    const I nw = status[LAQR0_NW];
    const I jw = status[LAQR0_JW];
    const I kwtop = status[LAQR0_KWTOP];
    const T s = statusT[LAQR0_SPIKE_IN];
    const I kv = n - nw + 1;
    const I kt = nw + 1;
    const I kwv = nw + 2;
    I ns, nd;
    bool update;
    T spike;
    hqr_sync();
    aed_core_block<BS, T, I, true>(n, jw, s, &h(kv, kt), ldh, &h(kv, 1), ldh, &h(kwv, 1),
                                   W + (kwtop - 1), ns, nd, update, spike, s_ired, s_sred, ibuf,
                                   sbuf, lds_ws, st4, stT4, sV4);
    if(tid == 0)
    {
        status[LAQR0_LS] = ns;
        status[LAQR0_LD] = nd;
        status[LAQR0_UPDATE] = update ? 1 : 0;
        status[LAQR0_DEFER] = 0;
        statusT[LAQR0_SPIKE_OUT] = spike;
    }
    hqr_sync();

    laqr0_part2_block<BS>(n, kbot, ndfl, nwmax, nsr, nsmax, H, ldh, W, status, statusT, s_ired, ibuf);
}

ROCSOLVER_END_NAMESPACE

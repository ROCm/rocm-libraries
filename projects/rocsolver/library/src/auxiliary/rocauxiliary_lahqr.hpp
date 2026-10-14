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

/** The routines below can run on the device, where they are executed by all the
    threads of a thread-block of BS threads, or (with BS = 1) on the host, by a single
    thread. HQR_TID returns the index of the calling thread, HQR_SYNC is a barrier of
    the thread-block (a no-op on the host), and HQR_RED(BS) is the size of the shared
    buffers used by the reductions. **/
#define HQR_RED(BS) ((BS) >= 32 ? (BS) / 32 : 1)

__host__ __device__ inline int hqr_tid()
{
#if defined(__HIP_DEVICE_COMPILE__)
    return hipThreadIdx_x;
#else
    return 0;
#endif
}

__host__ __device__ inline void hqr_sync()
{
#if defined(__HIP_DEVICE_COMPILE__)
    __syncthreads();
#endif
}

/*
 * ===========================================================================
 *    Device versions of the LAPACK routines used by the complex Hessenberg
 *    QR algorithm (ZLAHQR): machine constants (xLAMCH), xLAPY3, xLADIV,
 *    complex square root, and ZLARFG for small vectors. They are executed
 *    by a single thread (or redundantly by all the threads of a block).
 * ===========================================================================
 */

// xLAMCH('S'): safe minimum
template <typename S>
__host__ __device__ constexpr S hqr_safmin()
{
    return std::numeric_limits<S>::min();
}

// xLAMCH('E'): relative machine precision (rounding)
template <typename S>
__host__ __device__ constexpr S hqr_eps()
{
    return std::numeric_limits<S>::epsilon() / 2;
}

// xLAMCH('P'): eps * base
template <typename S>
__host__ __device__ constexpr S hqr_ulp()
{
    return std::numeric_limits<S>::epsilon();
}

// |Re(z)| + |Im(z)|
template <typename T, typename S = decltype(std::real(T{}))>
__host__ __device__ inline S hqr_cabs1(const T z)
{
    return std::abs(z.real()) + std::abs(z.imag());
}

/** HQR_CDIVR returns a / b; when a is complex and b real, it divides each component by b, as
    Fortran does. (rocblas_complex_num has no complex / real operator: b is converted to a complex
    number, and the complex division forms 1 / b, which overflows for subnormal b.) **/
template <typename A, typename B>
__host__ __device__ inline A hqr_cdivr(const A a, const B b)
{
    if constexpr(rocblas_is_complex<A> && !rocblas_is_complex<B>)
        return A(a.real() / b, a.imag() / b);
    else
        return a / b;
}

/** HQR_LAPY3 computes sqrt(x^2 + y^2 + z^2) avoiding unnecessary overflow
    (LAPACK xLAPY3). **/
template <typename S>
__host__ __device__ S hqr_lapy3(const S x, const S y, const S z)
{
    S xabs = std::abs(x);
    S yabs = std::abs(y);
    S zabs = std::abs(z);
    S w = std::max(xabs, std::max(yabs, zabs));
    if(w == 0 || w > std::numeric_limits<S>::max())
        // (w can be zero for max(0,nan,0); adding the three entries keeps the NaN)
        return xabs + yabs + zabs;
    else
        return w
            * std::sqrt((xabs / w) * (xabs / w) + (yabs / w) * (yabs / w) + (zabs / w) * (zabs / w));
}

/** HQR_LADIV2, HQR_LADIV1 and HQR_LADIV perform the robust complex division
    (a + ib) / (c + id) = p + iq of LAPACK xLADIV (Baudin & Smith). **/
template <typename S>
__host__ __device__ S hqr_ladiv2(const S a, const S b, const S c, const S d, const S r, const S t)
{
    if(r != 0)
    {
        S br = b * r;
        if(br != 0)
            return (a + br) * t;
        else
            return a * t + (b * t) * r;
    }
    else
        return (a + d * (b / c)) * t;
}

template <typename S>
__host__ __device__ void hqr_ladiv1(S a, const S b, const S c, const S d, S& p, S& q)
{
    S r = d / c;
    S t = S(1) / (c + d * r);
    p = hqr_ladiv2(a, b, c, d, r, t);
    a = -a;
    q = hqr_ladiv2(b, a, c, d, r, t);
}

template <typename S>
__host__ __device__ void hqr_ladiv(const S a, const S b, const S c, const S d, S& p, S& q)
{
    const S half = S(0.5);
    const S two = S(2);
    const S bs = S(2);

    S aa = a;
    S bb = b;
    S cc = c;
    S dd = d;
    S ab = std::max(std::abs(a), std::abs(b));
    S cd = std::max(std::abs(c), std::abs(d));
    S s = S(1);

    const S ov = std::numeric_limits<S>::max();
    const S un = hqr_safmin<S>();
    const S eps = hqr_eps<S>();
    const S be = bs / (eps * eps);

    if(ab >= half * ov)
    {
        aa = half * aa;
        bb = half * bb;
        s = two * s;
    }
    if(cd >= half * ov)
    {
        cc = half * cc;
        dd = half * dd;
        s = half * s;
    }
    if(ab <= un * bs / eps)
    {
        aa = aa * be;
        bb = bb * be;
        s = s / be;
    }
    if(cd <= un * bs / eps)
    {
        cc = cc * be;
        dd = dd * be;
        s = s * be;
    }
    if(std::abs(d) <= std::abs(c))
    {
        hqr_ladiv1(aa, bb, cc, dd, p, q);
    }
    else
    {
        hqr_ladiv1(bb, aa, dd, cc, p, q);
        q = -q;
    }
    p = p * s;
    q = q * s;
}

/** HQR_ZLADIV returns x / y computed with xLADIV (LAPACK ZLADIV). **/
template <typename T>
__host__ __device__ T hqr_zladiv(const T x, const T y)
{
    using S = decltype(std::real(T{}));
    S zr, zi;
    hqr_ladiv(x.real(), x.imag(), y.real(), y.imag(), zr, zi);
    return T(zr, zi);
}

/** HQR_CSQRT returns the principal square root of the complex number z
    (the Fortran intrinsic SQRT for complex arguments). **/
template <typename T>
__host__ __device__ T hqr_csqrt(const T z)
{
    using S = decltype(std::real(T{}));
    S x = z.real();
    S y = z.imag();
    if(x == 0 && y == 0)
        return T(0, y);

    S t = std::sqrt(S(0.5) * std::abs(x) + S(0.5) * std::hypot(x, y));
    if(x >= 0)
        return T(t, y / (2 * t));
    else
        return T(std::abs(y) / (2 * t), std::copysign(t, y));
}

/** HQR_LARFG generates an elementary reflector H of order N such that
    H^H * [alpha; x] = [beta; 0], with beta real (LAPACK ZLARFG with incx = 1).
    x has N-1 entries. On exit, alpha = beta and x is overwritten with the
    Householder vector v (with v(1) = 1 implicit). **/
template <int N, typename T>
__host__ __device__ void hqr_larfg(T& alpha, T* x, T& tau)
{
    using S = decltype(std::real(T{}));
    static_assert(N == 2 || N == 3, "hqr_larfg only supports vectors of 2 or 3 entries");

    auto xnorm2 = [&]() -> S {
        if constexpr(N == 2)
            return std::abs(x[0]);
        else
            return std::hypot(std::abs(x[0]), std::abs(x[1]));
    };

    S xnorm = xnorm2();
    S alphr = alpha.real();
    S alphi = alpha.imag();

    if(xnorm == 0 && alphi == 0)
    {
        // H = I
        tau = T(0);
        return;
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
            for(int j = 0; j < N - 1; j++)
                x[j] = rsafmn * x[j];
            beta = beta * rsafmn;
            alphi = alphi * rsafmn;
            alphr = alphr * rsafmn;
        } while(std::abs(beta) < safmin && knt < 20);

        // new beta is at most 1, at least safmin
        xnorm = xnorm2();
        alpha = T(alphr, alphi);
        beta = -std::copysign(hqr_lapy3(alphr, alphi, xnorm), alphr);
    }
    tau = T((beta - alphr) / beta, -alphi / beta);
    alpha = hqr_zladiv(T(1), alpha - beta);
    for(int j = 0; j < N - 1; j++)
        x[j] = alpha * x[j];

    // if alpha is subnormal, it may lose relative accuracy
    for(int j = 0; j < knt; j++)
        beta = beta * safmin;
    alpha = T(beta);
}

/** HQR_LARTG generates a plane rotation with real cosine and complex sine such that
        [  c         s ] [ f ]   [ r ]
        [ -conj(s)   c ] [ g ] = [ 0 ]
    using the safe-scaling algorithm of LAPACK ZLARTG (version 3.10 and later). **/
template <typename T, typename S = decltype(std::real(T{}))>
__host__ __device__ void hqr_lartg(const T f, const T g, S& c, T& s, T& r)
{
    const S zero = 0;
    const S one = 1;
    // (LAPACK's la_constants: safmin = radix^max(minexponent-1, 1-maxexponent))
    const S safmin = std::numeric_limits<S>::min();
    const S safmax = one / safmin;
    auto abssq = [](const T t) -> S { return t.real() * t.real() + t.imag() * t.imag(); };

    const S rtmin = std::sqrt(safmin);

    if(g == T(0))
    {
        c = one;
        s = T(0);
        r = f;
    }
    else if(f == T(0))
    {
        c = zero;
        if(g.real() == zero)
        {
            S rr = std::abs(g.imag());
            r = T(rr);
            s = hqr_cdivr(conj(g), rr);
        }
        else if(g.imag() == zero)
        {
            S rr = std::abs(g.real());
            r = T(rr);
            s = hqr_cdivr(conj(g), rr);
        }
        else
        {
            S g1 = std::max(std::abs(g.real()), std::abs(g.imag()));
            S rtmax = std::sqrt(safmax / 2);
            if(g1 > rtmin && g1 < rtmax)
            {
                S g2 = abssq(g);
                S d = std::sqrt(g2);
                s = hqr_cdivr(conj(g), d);
                r = T(d);
            }
            else
            {
                S u = std::min(safmax, std::max(safmin, g1));
                T gs = hqr_cdivr(g, u);
                S g2 = abssq(gs);
                S d = std::sqrt(g2);
                s = hqr_cdivr(conj(gs), d);
                r = T(d * u);
            }
        }
    }
    else
    {
        S f1 = std::max(std::abs(f.real()), std::abs(f.imag()));
        S g1 = std::max(std::abs(g.real()), std::abs(g.imag()));
        S rtmax = std::sqrt(safmax / 4);
        if(f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax)
        {
            S f2 = abssq(f);
            S g2 = abssq(g);
            S h2 = f2 + g2;
            if(f2 >= h2 * safmin)
            {
                c = std::sqrt(f2 / h2);
                r = hqr_cdivr(f, c);
                rtmax = rtmax * 2;
                if(f2 > rtmin && h2 < rtmax)
                    s = conj(g) * hqr_cdivr(f, std::sqrt(f2 * h2));
                else
                    s = conj(g) * hqr_cdivr(r, h2);
            }
            else
            {
                S d = std::sqrt(f2 * h2);
                c = f2 / d;
                if(c >= safmin)
                    r = hqr_cdivr(f, c);
                else
                    r = (h2 / d) * f;
                s = conj(g) * hqr_cdivr(f, d);
            }
        }
        else
        {
            S u = std::min(safmax, std::max(safmin, std::max(f1, g1)));
            T gs = hqr_cdivr(g, u);
            S g2 = abssq(gs);
            S w, f2, h2;
            T fs;
            if(f1 / u < rtmin)
            {
                S v = std::min(safmax, std::max(safmin, f1));
                w = v / u;
                fs = hqr_cdivr(f, v);
                f2 = abssq(fs);
                h2 = f2 * (w * w) + g2;
            }
            else
            {
                w = one;
                fs = hqr_cdivr(f, u);
                f2 = abssq(fs);
                h2 = f2 + g2;
            }
            if(f2 >= h2 * safmin)
            {
                c = std::sqrt(f2 / h2);
                r = hqr_cdivr(fs, c);
                rtmax = rtmax * 2;
                if(f2 > rtmin && h2 < rtmax)
                    s = conj(gs) * hqr_cdivr(fs, std::sqrt(f2 * h2));
                else
                    s = conj(gs) * hqr_cdivr(r, h2);
            }
            else
            {
                S d = std::sqrt(f2 * h2);
                c = f2 / d;
                if(c >= safmin)
                    r = hqr_cdivr(fs, c);
                else
                    r = (h2 / d) * fs;
                s = conj(gs) * hqr_cdivr(fs, d);
            }
            c = c * w;
            r = u * r;
        }
    }
}

/** HQR_ROT applies a plane rotation with real cosine and complex sine to the pair
    (x, y) (LAPACK ZROT): x <- c*x + s*y, y <- c*y - conj(s)*x. **/
template <typename T, typename S>
__host__ __device__ inline void hqr_rot(T& x, T& y, const S c, const T s)
{
    T stemp = c * x + s * y;
    y = c * y - conj(s) * x;
    x = stemp;
}

/** HQR_BLOCK_MAX returns the maximum of v over the thread-block to all the threads.
    It uses one of two shared buffers alternately (selected by buf), so that
    one barrier per call is enough. **/
template <int BS, typename I>
__host__ __device__ I hqr_block_max(I v, I (*s_red)[HQR_RED(BS)], int& buf)
{
    static_assert(BS == 1 || BS % 64 == 0, "BS must be 1 or a multiple of 64");
#if defined(__HIP_DEVICE_COMPILE__)
    const int lane = hqr_tid() % warpSize;
    const int wave = hqr_tid() / warpSize;
    const int nwaves = BS / warpSize;

    for(int offset = warpSize / 2; offset > 0; offset /= 2)
        v = std::max(v, I(__shfl_xor(v, offset)));
    if(lane == 0)
        s_red[buf][wave] = v;
    hqr_sync();
    v = s_red[buf][0];
    for(int w = 1; w < nwaves; w++)
        v = std::max(v, s_red[buf][w]);
    buf = 1 - buf;
    return v;
#else
    return v;
#endif
}

/** LAHQR_BLOCK computes the eigenvalues and, optionally, the Schur form and
    Schur vectors of the complex upper Hessenberg matrix H, using the
    single-shift QR algorithm of LAPACK ZLAHQR (version 3.12).

    It must be called by all the threads of a thread-block of BS threads, and
    returns info (0, or i > 0 if the algorithm failed to compute the i-th eigenvalue,
    as in LAPACK) to all the threads.

    The indices ilo, ihi, iloz and ihiz are 1-based, as in LAPACK. To make the
    comparison with the reference implementation straightforward, the code uses
    1-based indices throughout. The scalar computations (shifts, reflectors) are
    performed redundantly by all the threads, so that all of them take the same
    decisions; the updates of rows and columns of H and Z are distributed among
    the threads. s_red is a shared buffer of 2 x BS/32 integers. **/
template <int BS, typename T, typename I>
__host__ __device__ __forceinline__ I lahqr_block(const bool wantt,
                                                  const bool wantz,
                                                  const I n,
                                                  const I ilo,
                                                  const I ihi,
                                                  T* H,
                                                  const I ldh,
                                                  T* W,
                                                  const I iloz,
                                                  const I ihiz,
                                                  T* Z,
                                                  const I ldz,
                                                  I (*s_red)[HQR_RED(BS)],
                                                  int& buf)
{
    using S = decltype(std::real(T{}));

    const I tid = hqr_tid();
    auto h = [&](const I i, const I j) -> T& { return H[idx2D(i - 1, j - 1, ldh)]; };
    auto z = [&](const I i, const I j) -> T& { return Z[idx2D(i - 1, j - 1, ldz)]; };

    const S rzero = 0;
    const S half = S(0.5);
    const S dat1 = S(3) / S(4);
    const I kexsh = 10;

    // quick return if possible
    if(n == 0)
        return 0;
    if(ilo == ihi)
    {
        if(tid == 0)
            W[ilo - 1] = h(ilo, ilo);
        return 0;
    }

    // clear out the trash
    for(I j = ilo + tid; j <= ihi - 3; j += BS)
    {
        h(j + 2, j) = T(0);
        h(j + 3, j) = T(0);
    }
    if(tid == 0 && ilo <= ihi - 2)
        h(ihi, ihi - 2) = T(0);
    hqr_sync();

    // ensure that subdiagonal entries are real
    I jlo, jhi;
    if(wantt)
    {
        jlo = 1;
        jhi = n;
    }
    else
    {
        jlo = ilo;
        jhi = ihi;
    }
    for(I i = ilo + 1; i <= ihi; i++)
    {
        T hi = h(i, i - 1);
        if(hi.imag() != rzero)
        {
            // the following redundant normalization avoids problems with both
            // gradual and sudden underflow in abs(h(i,i-1))
            T sc = hqr_cdivr(hi, hqr_cabs1(hi));
            sc = hqr_cdivr(conj(sc), std::abs(sc));
            S habs = std::abs(hi);
            hqr_sync();

            if(tid == 0)
                h(i, i - 1) = T(habs);
            // row i is scaled by sc, and column i by conj(sc); h(i,i) is scaled
            // by both (in this order)
            for(I j = i + tid; j <= jhi; j += BS)
            {
                T v = sc * h(i, j);
                if(j == i)
                    v = conj(sc) * v;
                h(i, j) = v;
            }
            for(I j = jlo + tid; j <= std::min(jhi, i + 1); j += BS)
                if(j != i)
                    h(j, i) = conj(sc) * h(j, i);
            if(wantz)
                for(I j = iloz + tid; j <= ihiz; j += BS)
                    z(j, i) = conj(sc) * z(j, i);
            hqr_sync();
        }
    }

    const I nh = ihi - ilo + 1;

    // machine-dependent constants for the stopping criterion
    const S safmin = hqr_safmin<S>();
    const S ulp = hqr_ulp<S>();
    const S smlnum = safmin * (S(nh) / ulp);

    // i1 and i2 are the indices of the first row and last column of H to which
    // transformations must be applied. If eigenvalues only are being computed,
    // they are set inside the main loop.
    I i1 = 1;
    I i2 = n;

    // itmax is the number of QR iterations allowed (for each eigenvalue)
    const I itmax = 30 * std::max(I(10), nh);

    // kdefl counts the number of iterations since a deflation
    I kdefl = 0;

    // test for a negligible subdiagonal entry h(k,k-1)
    auto negligible = [&](const I k) -> bool {
        T hk = h(k, k - 1);
        if(hqr_cabs1(hk) <= smlnum)
            return true;
        S tst = hqr_cabs1(h(k - 1, k - 1)) + hqr_cabs1(h(k, k));
        if(tst == rzero)
        {
            if(k - 2 >= ilo)
                tst = tst + std::abs(h(k - 1, k - 2).real());
            if(k + 1 <= ihi)
                tst = tst + std::abs(h(k + 1, k).real());
        }
        // conservative small subdiagonal deflation criterion due to
        // Ahues & Tisseur (LAWN 122, 1997)
        if(std::abs(hk.real()) <= ulp * tst)
        {
            S ab = std::max(hqr_cabs1(hk), hqr_cabs1(h(k - 1, k)));
            S ba = std::min(hqr_cabs1(hk), hqr_cabs1(h(k - 1, k)));
            S aa = std::max(hqr_cabs1(h(k, k)), hqr_cabs1(h(k - 1, k - 1) - h(k, k)));
            S bb = std::min(hqr_cabs1(h(k, k)), hqr_cabs1(h(k - 1, k - 1) - h(k, k)));
            S s = aa + ab;
            if(ba * (ab / s) <= std::max(smlnum, ulp * (bb * (aa / s))))
                return true;
        }
        return false;
    };

    // The main loop begins here. i is the loop index and decreases from ihi
    // to ilo in steps of 1. Each iteration of the loop works with the active
    // submatrix in rows and columns l to i. Eigenvalues i+1 to ihi have already
    // converged. Either l = ilo, or h(l,l-1) is negligible so that the matrix splits.
    I i = ihi;
    while(i >= ilo)
    {
        // Perform QR iterations on rows and columns ilo to i until a submatrix
        // of order 1 splits off at the bottom because a subdiagonal element
        // has become negligible.
        I l = ilo;
        bool converged = false;
        for(I its = 0; its <= itmax; its++)
        {
            // Look for a single small subdiagonal element
            // (the largest k in l+1:i such that h(k,k-1) is negligible, or l).
            I kfound = 0;
            for(I k = i - tid; k >= l + 1; k -= BS)
            {
                if(negligible(k))
                {
                    kfound = k;
                    break;
                }
            }
            kfound = hqr_block_max<BS>(kfound, s_red, buf);
            l = (kfound > 0) ? kfound : l;

            if(l > ilo)
            {
                // h(l,l-1) is negligible
                if(tid == 0)
                    h(l, l - 1) = T(0);
            }
            hqr_sync();

            // exit from loop if a submatrix of order 1 has split off
            if(l >= i)
            {
                converged = true;
                break;
            }
            kdefl++;

            // Now the active submatrix is in rows and columns l to i. If
            // eigenvalues only are being computed, only the active submatrix
            // need be transformed.
            if(!wantt)
            {
                i1 = l;
                i2 = i;
            }

            T t;
            if(kdefl % (2 * kexsh) == 0)
            {
                // exceptional shift
                S s = dat1 * std::abs(h(i, i - 1).real());
                t = s + h(i, i);
            }
            else if(kdefl % kexsh == 0)
            {
                // exceptional shift
                S s = dat1 * std::abs(h(l + 1, l).real());
                t = s + h(l, l);
            }
            else
            {
                // Wilkinson's shift
                t = h(i, i);
                T u = hqr_csqrt(h(i - 1, i)) * hqr_csqrt(h(i, i - 1));
                S s = hqr_cabs1(u);
                if(s != rzero)
                {
                    T x = half * (h(i - 1, i - 1) - t);
                    S sx = hqr_cabs1(x);
                    s = std::max(s, hqr_cabs1(x));
                    T xs = hqr_cdivr(x, s);
                    T us = hqr_cdivr(u, s);
                    T y = s * hqr_csqrt(xs * xs + us * us);
                    if(sx > rzero)
                    {
                        T xsx = hqr_cdivr(x, sx);
                        if(xsx.real() * y.real() + xsx.imag() * y.imag() < rzero)
                            y = -y;
                    }
                    t = t - u * hqr_zladiv(u, x + y);
                }
            }

            // Look for two consecutive small subdiagonal elements: determine the
            // effect of starting the single-shift QR iteration at row m, and see
            // if this would make h(m,m-1) negligible (the largest such m in
            // l+1:i-1, or l).
            auto start_vector = [&](const I m, T& v1, S& v2) {
                T h11 = h(m, m);
                T h11s = h11 - t;
                S h21 = h(m + 1, m).real();
                S s = hqr_cabs1(h11s) + std::abs(h21);
                v1 = hqr_cdivr(h11s, s);
                v2 = hqr_cdivr(h21, s);
            };
            I mfound = 0;
            for(I m = i - 1 - tid; m >= l + 1; m -= BS)
            {
                T h11s;
                S h21;
                start_vector(m, h11s, h21);
                T h11 = h(m, m);
                T h22 = h(m + 1, m + 1);
                S h10 = h(m, m - 1).real();
                if(std::abs(h10) * std::abs(h21)
                   <= ulp * (hqr_cabs1(h11s) * (hqr_cabs1(h11) + hqr_cabs1(h22))))
                {
                    mfound = m;
                    break;
                }
            }
            mfound = hqr_block_max<BS>(mfound, s_red, buf);
            const I m = (mfound > 0) ? mfound : l;

            T v[2];
            {
                S v2;
                start_vector(m, v[0], v2);
                v[1] = T(v2);
            }
            // all the threads must have read h(m,m) and h(m+1,m) before the first
            // reflection is applied (it updates rows m and m+1 from column m on)
            hqr_sync();

            // single-shift QR step
            for(I k = m; k <= i - 1; k++)
            {
                // The first iteration of this loop determines a reflection G from
                // the vector v and applies it from left and right to H, thus
                // creating a nonzero bulge below the subdiagonal. Each subsequent
                // iteration determines a reflection G to restore the Hessenberg
                // form in the (k-1)th column, and thus chases the bulge one step
                // toward the bottom of the active submatrix. v(2) is always real
                // before the call to larfg, and hence after the call t2 (= t1*v(2))
                // is also real.
                if(k > m)
                {
                    v[0] = h(k, k - 1);
                    v[1] = h(k + 1, k - 1);
                }
                T t1;
                hqr_larfg<2>(v[0], v + 1, t1);
                const T v2 = v[1];
                const S t2 = (t1 * v2).real();

                // apply G from the left to transform the rows of the matrix in
                // columns k to i2
                for(I j = k + tid; j <= i2; j += BS)
                {
                    T sum = conj(t1) * h(k, j) + t2 * h(k + 1, j);
                    h(k, j) = h(k, j) - sum;
                    h(k + 1, j) = h(k + 1, j) - sum * v2;
                }
                hqr_sync();

                // (column k-1 is not used by the updates below, and all the threads
                // have read it)
                if(k > m && tid == 0)
                {
                    h(k, k - 1) = v[0];
                    h(k + 1, k - 1) = T(0);
                }

                // apply G from the right to transform the columns of the matrix in
                // rows i1 to min(k+2,i)
                for(I j = i1 + tid; j <= std::min(k + 2, i); j += BS)
                {
                    T sum = t1 * h(j, k) + t2 * h(j, k + 1);
                    h(j, k) = h(j, k) - sum;
                    h(j, k + 1) = h(j, k + 1) - sum * conj(v2);
                }

                // accumulate transformations in the matrix Z
                if(wantz)
                {
                    for(I j = iloz + tid; j <= ihiz; j += BS)
                    {
                        T sum = t1 * z(j, k) + t2 * z(j, k + 1);
                        z(j, k) = z(j, k) - sum;
                        z(j, k + 1) = z(j, k + 1) - sum * conj(v2);
                    }
                }
                hqr_sync();

                if(k == m && m > l)
                {
                    // If the QR step was started at row m > l because two
                    // consecutive small subdiagonals were found, then extra
                    // scaling must be performed to ensure that h(m,m-1) remains real.
                    T temp = T(1) - t1;
                    temp = hqr_cdivr(temp, std::abs(temp));
                    if(tid == 0)
                    {
                        h(m + 1, m) = h(m + 1, m) * conj(temp);
                        if(m + 2 <= i)
                            h(m + 2, m + 1) = h(m + 2, m + 1) * temp;
                    }
                    // For j = m:i, j != m+1, row j (columns j+1:i2) is scaled by temp
                    // and column j (rows i1:j-1) by conj(temp). An entry (r,c) with
                    // r < c both in the set is scaled first by temp and then by conj(temp).
                    for(I r = m; r <= i; r++)
                    {
                        if(r == m + 1)
                            continue;
                        for(I c = r + 1 + tid; c <= i2; c += BS)
                        {
                            T val = temp * h(r, c);
                            if(c >= m && c <= i && c != m + 1)
                                val = conj(temp) * val;
                            h(r, c) = val;
                        }
                    }
                    for(I c = m; c <= i; c++)
                    {
                        if(c == m + 1)
                            continue;
                        for(I r = i1 + tid; r <= c - 1; r += BS)
                            if(r < m || r > i || r == m + 1)
                                h(r, c) = conj(temp) * h(r, c);
                        if(wantz)
                            for(I r = iloz + tid; r <= ihiz; r += BS)
                                z(r, c) = conj(temp) * z(r, c);
                    }
                    hqr_sync();
                }
            }

            // ensure that h(i,i-1) is real
            T temp = h(i, i - 1);
            if(temp.imag() != rzero)
            {
                S rtemp = std::abs(temp);
                temp = hqr_cdivr(temp, rtemp);
                for(I j = i + 1 + tid; j <= i2; j += BS)
                    h(i, j) = conj(temp) * h(i, j);
                for(I j = i1 + tid; j <= i - 1; j += BS)
                    h(j, i) = temp * h(j, i);
                if(wantz)
                    for(I j = iloz + tid; j <= ihiz; j += BS)
                        z(j, i) = temp * z(j, i);
                hqr_sync();
                if(tid == 0)
                    h(i, i - 1) = T(rtemp);
                hqr_sync();
            }
        }

        if(!converged)
        {
            // failure to converge in remaining number of iterations
            return i;
        }

        // h(i,i-1) is negligible: one eigenvalue has converged
        if(tid == 0)
            W[i - 1] = h(i, i);
        // reset deflation counter
        kdefl = 0;

        // return to start of the main loop with new value of i
        i = l - 1;
    }

    return 0;
}

/*
 * ===========================================================================
 *    LAHQR_LDS_BLOCK: ZLAHQR for a small matrix (n <= HQR_LDS_NMAX) with the Schur
 *    form and Schur vectors (wantt, wantz, ilo = iloz = 1, ihi = ihiz = n), as used
 *    by the aggressive early deflation. It follows lahqr_block, but:
 *    - the upper Hessenberg matrix (and the bulge) is kept in shared memory, packed
 *      by columns (column j holds rows 1:min(j+2,n));
 *    - the 2-element reflections are computed directly when no scaling is needed;
 *    - the updates of Z are deferred to the end of each QR sweep, when each thread
 *      applies the transformations of the sweep to one row of Z.
 *    The QR iteration is a chain of small dependent steps; these changes shorten the
 *    latency of each step. All the threads of the block (BS >= HQR_LDS_NMAX) must call
 *    it. The shared workspace ws must hold HQR_LDS_WS_SIZE entries of type T.
 * ===========================================================================
 */
#define HQR_LDS_NMAX 64
// packed matrix, 2*HQR_LDS_NMAX transformations, and HQR_LDS_NMAX + 2 integers (column
// offsets and a reduction variable; 40 entries of at least 8 bytes). The packed matrix of
// order n keeps min(j + 2, n) entries of column j: n(n+1)/2 + 2n - 3 entries (rounded up to
// a multiple of 8).
#define HQR_LDS_PACKED_SIZE \
    (((HQR_LDS_NMAX * (HQR_LDS_NMAX + 1) / 2 + 2 * HQR_LDS_NMAX - 3 + 7) / 8) * 8)
#define HQR_LDS_WS_SIZE (HQR_LDS_PACKED_SIZE + 2 * HQR_LDS_NMAX + 40)

/** HQR_LARFG2_FAST computes a 2-element reflection like hqr_larfg<2>, directly when
    alpha and x are in a range where no scaling is needed. **/
template <typename T>
__device__ inline T hqr_larfg2_fast(T& alpha, T& x)
{
    using S = decltype(std::real(T{}));
    const S xr = x.real(), xi = x.imag();
    const S ar = alpha.real(), ai = alpha.imag();
    const S xn2 = xr * xr + xi * xi;
    if(xr == 0 && xi == 0 && ai == 0)
        return T(0);
    const S big = std::is_same<S, double>::value ? S(1e100) : S(1e15);
    const S m = std::max(std::max(std::abs(ar), std::abs(ai)), std::sqrt(xn2));
    // (|x|^2 underflows for a tiny x: then the scaled version, as xLARFG)
    if(!(m < big && m > S(1) / big) || xn2 < std::numeric_limits<S>::min())
    {
        T tau;
        hqr_larfg<2>(alpha, &x, tau);
        return tau;
    }
    const S beta = -std::copysign(std::sqrt(ar * ar + ai * ai + xn2), ar);
    const T tau = T((beta - ar) / beta, -ai / beta);
    // x <- x / (alpha - beta)
    const S dr = ar - beta;
    const S rd = S(1) / (dr * dr + ai * ai);
    x = x * T(dr * rd, -ai * rd);
    alpha = T(beta);
    return tau;
}

template <int BS, typename T, typename I>
__device__ __forceinline__ I
    lahqr_lds_block(const I n, T* Hg, const I ldh, T* Wg, T* Zg, const I ldz, T* ws)
{
    static_assert(BS >= HQR_LDS_NMAX, "lahqr_lds_block needs at least HQR_LDS_NMAX threads");
    static_assert((HQR_LDS_NMAX + 2) * sizeof(int) <= 40 * sizeof(T),
                  "the integers must fit in the last 40 entries of the workspace");
    using S = decltype(std::real(T{}));
    const I tid = hipThreadIdx_x;

    T* Hs = ws;
    T* rt1 = ws + HQR_LDS_PACKED_SIZE;
    T* rv2 = rt1 + HQR_LDS_NMAX;
    int* offs = reinterpret_cast<int*>(rv2 + HQR_LDS_NMAX);
    int* s_int = offs + HQR_LDS_NMAX + 1;

    if(tid == 0)
    {
        int sz = 0;
        for(int j = 1; j <= n; j++)
        {
            offs[j] = sz - 1; // h(i,j) is Hs[offs[j] + i]
            sz += std::min(j + 2, int(n));
        }
    }
    __syncthreads();
    auto h = [&](const I i, const I j) -> T& { return Hs[offs[j] + i]; };
    auto z = [&](const I i, const I j) -> T& { return Zg[idx2D(i - 1, j - 1, ldz)]; };

    // load the Hessenberg part (the entries two positions below the diagonal are
    // cleared, as in lahqr_block)
    for(I j = 1; j <= n; j++)
        for(I i = 1 + tid; i <= std::min(j + 2, n); i += BS)
            h(i, j) = (i <= j + 1) ? Hg[idx2D(i - 1, j - 1, ldh)] : T(0);
    __syncthreads();

    const S rzero = 0;
    const S half = S(0.5);
    const S dat1 = S(3) / S(4);
    const I kexsh = 10;
    const I ilo = 1, ihi = n, i1 = 1, i2 = n;

    // ensure that subdiagonal entries are real
    for(I i = ilo + 1; i <= ihi; i++)
    {
        T hi = h(i, i - 1);
        if(hi.imag() != rzero)
        {
            T sc = hqr_cdivr(hi, hqr_cabs1(hi));
            sc = hqr_cdivr(conj(sc), std::abs(sc));
            S habs = std::abs(hi);
            __syncthreads();
            for(I j = i + tid; j <= i2; j += BS)
            {
                T v = sc * h(i, j);
                if(j == i)
                    v = conj(sc) * v;
                h(i, j) = v;
            }
            for(I j = i1 + tid; j <= std::min(i2, i + 1); j += BS)
                if(j != i)
                    h(j, i) = conj(sc) * h(j, i);
            for(I j = 1 + tid; j <= n; j += BS)
                z(j, i) = conj(sc) * z(j, i);
            __syncthreads();
            if(tid == 0)
                h(i, i - 1) = T(habs);
            __syncthreads();
        }
    }

    const I nh = ihi - ilo + 1;
    const S safmin = hqr_safmin<S>();
    const S ulp = hqr_ulp<S>();
    const S smlnum = safmin * (S(nh) / ulp);
    const I itmax = 30 * std::max(I(10), nh);
    I kdefl = 0;

    auto negligible = [&](const I k) -> bool {
        T hk = h(k, k - 1);
        if(hqr_cabs1(hk) <= smlnum)
            return true;
        S tst = hqr_cabs1(h(k - 1, k - 1)) + hqr_cabs1(h(k, k));
        if(tst == rzero)
        {
            if(k - 2 >= ilo)
                tst = tst + std::abs(h(k - 1, k - 2).real());
            if(k + 1 <= ihi)
                tst = tst + std::abs(h(k + 1, k).real());
        }
        if(std::abs(hk.real()) <= ulp * tst)
        {
            S ab = std::max(hqr_cabs1(hk), hqr_cabs1(h(k - 1, k)));
            S ba = std::min(hqr_cabs1(hk), hqr_cabs1(h(k - 1, k)));
            S aa = std::max(hqr_cabs1(h(k, k)), hqr_cabs1(h(k - 1, k - 1) - h(k, k)));
            S bb = std::min(hqr_cabs1(h(k, k)), hqr_cabs1(h(k - 1, k - 1) - h(k, k)));
            S s = aa + ab;
            if(ba * (ab / s) <= std::max(smlnum, ulp * (bb * (aa / s))))
                return true;
        }
        return false;
    };

    I i = ihi;
    I result = 0;
    while(i >= ilo)
    {
        I l = ilo;
        bool converged = false;
        for(I its = 0; its <= itmax; its++)
        {
            // the largest k in l+1:i such that h(k,k-1) is negligible, or l
            if(tid == 0)
                *s_int = 0;
            __syncthreads();
            {
                const I k = i - tid;
                if(tid < HQR_LDS_NMAX && k >= l + 1 && negligible(k))
                    atomicMax(s_int, int(k));
            }
            __syncthreads();
            const I kfound = *s_int;
            l = (kfound > 0) ? kfound : l;
            __syncthreads();
            if(l > ilo && tid == 0)
                h(l, l - 1) = T(0);
            __syncthreads();
            if(l >= i)
            {
                converged = true;
                break;
            }
            kdefl++;

            T t;
            if(kdefl % (2 * kexsh) == 0)
            {
                S s = dat1 * std::abs(h(i, i - 1).real());
                t = s + h(i, i);
            }
            else if(kdefl % kexsh == 0)
            {
                S s = dat1 * std::abs(h(l + 1, l).real());
                t = s + h(l, l);
            }
            else
            {
                t = h(i, i);
                T u = hqr_csqrt(h(i - 1, i)) * hqr_csqrt(h(i, i - 1));
                S s = hqr_cabs1(u);
                if(s != rzero)
                {
                    T x = half * (h(i - 1, i - 1) - t);
                    S sx = hqr_cabs1(x);
                    s = std::max(s, hqr_cabs1(x));
                    T xs = hqr_cdivr(x, s);
                    T us = hqr_cdivr(u, s);
                    T y = s * hqr_csqrt(xs * xs + us * us);
                    if(sx > rzero)
                    {
                        T xsx = hqr_cdivr(x, sx);
                        if(xsx.real() * y.real() + xsx.imag() * y.imag() < rzero)
                            y = -y;
                    }
                    t = t - u * hqr_zladiv(u, x + y);
                }
            }

            auto start_vector = [&](const I m, T& v1, S& v2) {
                T h11s = h(m, m) - t;
                S h21 = h(m + 1, m).real();
                S s = hqr_cabs1(h11s) + std::abs(h21);
                v1 = hqr_cdivr(h11s, s);
                v2 = hqr_cdivr(h21, s);
            };
            // the largest m in l+1:i-1 where two consecutive small subdiagonal entries are
            // found, or l
            if(tid == 0)
                *s_int = 0;
            __syncthreads();
            {
                const I m = i - 1 - tid;
                if(tid < HQR_LDS_NMAX && m >= l + 1)
                {
                    T h11s;
                    S h21;
                    start_vector(m, h11s, h21);
                    S h10 = h(m, m - 1).real();
                    if(std::abs(h10) * std::abs(h21) <= ulp
                           * (hqr_cabs1(h11s) * (hqr_cabs1(h(m, m)) + hqr_cabs1(h(m + 1, m + 1)))))
                        atomicMax(s_int, int(m));
                }
            }
            __syncthreads();
            const I m = (*s_int > 0) ? I(*s_int) : l;

            T v0, v1;
            {
                S v2r;
                start_vector(m, v0, v2r);
                v1 = T(v2r);
            }
            __syncthreads();

            // single-shift QR step (the transformations of Z are stored)
            bool zscale = false;
            T zsc = T(1);
            for(I k = m; k <= i - 1; k++)
            {
                if(k > m)
                {
                    v0 = h(k, k - 1);
                    v1 = h(k + 1, k - 1);
                }
                const T t1 = hqr_larfg2_fast(v0, v1);
                const T v2 = v1;
                const S t2 = (t1 * v2).real();
                if(tid == 0)
                {
                    rt1[k - m] = t1;
                    rv2[k - m] = v2;
                }

                // rows k and k+1, columns k:i2
                {
                    const I j = k + tid;
                    if(j <= i2)
                    {
                        T hk = h(k, j), hk1 = h(k + 1, j);
                        T sum = conj(t1) * hk + t2 * hk1;
                        h(k, j) = hk - sum;
                        h(k + 1, j) = hk1 - sum * v2;
                    }
                }
                __syncthreads();
                if(k > m && tid == 0)
                {
                    h(k, k - 1) = v0;
                    h(k + 1, k - 1) = T(0);
                }
                // columns k and k+1, rows i1:min(k+2,i)
                {
                    const I j = i1 + tid;
                    if(j <= std::min(k + 2, i))
                    {
                        T hk = h(j, k), hk1 = h(j, k + 1);
                        T sum = t1 * hk + t2 * hk1;
                        h(j, k) = hk - sum;
                        h(j, k + 1) = hk1 - sum * conj(v2);
                    }
                }
                __syncthreads();

                if(k == m && m > l)
                {
                    // extra scaling to keep h(m,m-1) real (see lahqr_block)
                    T temp = T(1) - t1;
                    temp = hqr_cdivr(temp, std::abs(temp));
                    zscale = true;
                    zsc = conj(temp);
                    if(tid == 0)
                    {
                        h(m + 1, m) = h(m + 1, m) * conj(temp);
                        if(m + 2 <= i)
                            h(m + 2, m + 1) = h(m + 2, m + 1) * temp;
                    }
                    for(I r = m; r <= i; r++)
                    {
                        if(r == m + 1)
                            continue;
                        for(I c = r + 1 + tid; c <= i2; c += BS)
                        {
                            T val = temp * h(r, c);
                            if(c >= m && c <= i && c != m + 1)
                                val = conj(temp) * val;
                            h(r, c) = val;
                        }
                    }
                    for(I c = m; c <= i; c++)
                    {
                        if(c == m + 1)
                            continue;
                        for(I r = i1 + tid; r <= c - 1; r += BS)
                            if(r < m || r > i || r == m + 1)
                                h(r, c) = conj(temp) * h(r, c);
                    }
                    __syncthreads();
                }
            }

            // ensure that h(i,i-1) is real
            T tempf = h(i, i - 1);
            bool zfix = false;
            if(tempf.imag() != rzero)
            {
                S rtemp = std::abs(tempf);
                tempf = hqr_cdivr(tempf, rtemp);
                zfix = true;
                for(I j = i + 1 + tid; j <= i2; j += BS)
                    h(i, j) = conj(tempf) * h(i, j);
                for(I j = i1 + tid; j <= i - 1; j += BS)
                    h(j, i) = tempf * h(j, i);
                __syncthreads();
                if(tid == 0)
                    h(i, i - 1) = T(rtemp);
            }

            // deferred update of Z: row tid+1 gets the transformations k = m:i-1, the
            // scaling of columns m and m+2:i by zsc after the first one (if zscale), and
            // the scaling of column i by tempf (if zfix). The entries of the row are loaded
            // PF at a time.
            {
                const I r = 1 + tid;
                if(r <= n)
                {
                    constexpr int PF = 8;
                    T cur = z(r, m);
                    for(I kb = m; kb <= i - 1; kb += PF)
                    {
                        T nb[PF];
#pragma unroll
                        for(int q = 0; q < PF; q++)
                            if(kb + q <= i - 1)
                                nb[q] = z(r, kb + q + 1);
#pragma unroll
                        for(int q = 0; q < PF; q++)
                        {
                            const I k = kb + q;
                            if(k > i - 1)
                                break;
                            T nxt = nb[q];
                            if(zscale && k + 1 >= m + 2)
                                nxt = zsc * nxt;
                            const T t1 = rt1[k - m];
                            const T v2 = rv2[k - m];
                            const S t2 = (t1 * v2).real();
                            T sum = t1 * cur + t2 * nxt;
                            cur = cur - sum;
                            nxt = nxt - sum * conj(v2);
                            if(k == m && zscale)
                                cur = zsc * cur;
                            z(r, k) = cur;
                            cur = nxt;
                        }
                    }
                    if(zfix)
                        cur = tempf * cur;
                    z(r, i) = cur;
                }
            }
            __syncthreads();
        }

        if(!converged)
        {
            result = i;
            break;
        }
        if(tid == 0)
            Wg[i - 1] = h(i, i);
        kdefl = 0;
        i = l - 1;
    }

    // store the Schur form (upper Hessenberg part; zero below it)
    __syncthreads();
    for(I j = 1; j <= n; j++)
        for(I ii = 1 + tid; ii <= n; ii += BS)
            Hg[idx2D(ii - 1, j - 1, ldh)] = (ii <= j + 1) ? h(ii, j) : T(0);
    __syncthreads();
    return result;
}

ROCSOLVER_END_NAMESPACE

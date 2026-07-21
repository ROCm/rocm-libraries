/* **************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib_host_helpers.hpp"
#include "lib_macros.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

/*
 * ===========================================================================
 *    common location for host functions that reproduce LAPACK
 *    and BLAS functionality.
 * ===========================================================================
 */

static void call_lamch(char& cmach, float& eps)
{
    switch(cmach)
    {
    case 'E':
    case 'e': eps = std::numeric_limits<float>::epsilon(); return;
    case 'S':
    case 's': eps = std::numeric_limits<float>::min(); return;
    case 'B':
    case 'b': eps = FLT_RADIX; return;
    default: eps = std::numeric_limits<float>::min();
    }
}

static void call_lamch(char& cmach, double& eps)
{
    switch(cmach)
    {
    case 'E':
    case 'e': eps = std::numeric_limits<double>::epsilon(); return;
    case 'S':
    case 's': eps = std::numeric_limits<double>::min(); return;
    case 'B':
    case 'b': eps = FLT_RADIX; return;
    default: eps = std::numeric_limits<double>::min();
    }
}

template <typename T>
static void call_las2(T& f, T& g, T& h, T& ssmin, T& ssmax)
{
    T const zero = 0;
    T const one = 1;
    T const two = 2;

    T as, at, au, c, fa, fhmn, fhmx, ga, ha;

    auto square = [](auto x) { return (x * x); };

    fa = std::abs(f);
    ga = std::abs(g);
    ha = std::abs(h);
    fhmn = std::min(fa, ha);
    fhmx = std::max(fa, ha);
    if(fhmn == zero)
    {
        ssmin = zero;
        if(fhmx == zero)
        {
            ssmax = ga;
        }
        else
        {
            // ssmax = max( fhmx, ga )*sqrt( one+ ( min( fhmx, ga ) / max( fhmx, ga ) )**2 );
            ssmax = std::max(fhmx, ga)
                * std::sqrt(one + square(std::min(fhmx, ga) / std::max(fhmx, ga)));
        }
    }
    else
    {
        if(ga < fhmx)
        {
            as = one + fhmn / fhmx;
            at = (fhmx - fhmn) / fhmx;
            au = square(ga / fhmx);
            c = two / (std::sqrt(as * as + au) + std::sqrt(at * at + au));
            ssmin = fhmn * c;
            ssmax = fhmx / c;
        }
        else
        {
            au = fhmx / ga;
            if(au == zero)
            {
                //
                //               avoid possible harmful underflow if exponent range
                //               asymmetric (true ssmin may not underflow even if
                //               au underflows)
                //
                ssmin = (fhmn * fhmx) / ga;
                ssmax = ga;
            }
            else
            {
                as = one + fhmn / fhmx;
                at = (fhmx - fhmn) / fhmx;
                // c = one / ( sqrt( one+( as*au )**2 )+ sqrt( one+( at*au )**2 ) );
                c = one / (std::sqrt(one + square(as * au)) + std::sqrt(one + square(at * au)));
                ssmin = (fhmn * c) * au;
                ssmin = ssmin + ssmin;
                ssmax = ga / (c + c);
            }
        }
    }
}

template <typename T, typename S>
static void call_lartg(T& f, T& g, S& cs, T& sn, T& r)
{
    // ------------------------------------------------------
    // lartg generates a plane rotation so that
    // [  cs  sn ] * [ f ] = [ r ]
    // [ -sn  cs ]   [ g ]   [ 0 ]
    //
    // where cs * cs + abs(sn)*abs(sn) == 1
    // if g == 0, then cs == 1, sn == 0
    // if f == 0, then cs = 0, sn is chosen so that r is real
    // ------------------------------------------------------

    auto dble = [](auto z) { return (static_cast<double>(real_part(z))); };
    auto dimag = [](auto z) { return (static_cast<double>(imag_part(z))); };
    auto disnan = [](auto x) -> bool { return (isnan(x)); };
    auto dcmplx = [](auto x, auto y) -> T {
        bool constexpr is_complex_type = rocblas_is_complex<T>;

        if constexpr(is_complex_type)
        {
            return (T(x, y));
        }
        else
        {
            return (T(x));
        };
    };
    auto dconjg = [&](auto z) { return (dcmplx(dble(z), -dimag(z))); };

    auto square = [](auto x) { return (x * x); };

    auto abs1 = [&](auto ff) { return (std::max(std::abs(dble(ff)), std::abs(dimag(ff)))); };
    auto abssq = [&](auto ff) { return (square(dble(ff)) + square(dimag(ff))); };

    // -----------------------------------------
    // compute  sqrt( x * x + y * y )
    // without unnecessary overflow or underflow
    // -----------------------------------------
    auto dlapy2 = [&](auto x, auto y) {
        auto const one = 1;
        auto const zero = 0;

        auto ddlapy2 = x;
        bool const x_is_nan = disnan(x);
        bool const y_is_nan = disnan(y);
        if(x_is_nan)
            ddlapy2 = x;
        if(y_is_nan)
            ddlapy2 = y;

        if(!(x_is_nan || y_is_nan))
        {
            auto const xabs = std::abs(x);
            auto const yabs = std::abs(y);
            auto const w = std::max(xabs, yabs);
            auto const z = std::min(xabs, yabs);
            if(z == zero)
            {
                ddlapy2 = w;
            }
            else
            {
                ddlapy2 = w * std::sqrt(one + square(z / w));
            }
        }
        return (ddlapy2);
    };

    char cmach = 'E';
    S const zero = 0;
    S const one = 1;
    S const two = 2;
    T const czero = 0;

    bool has_work;
    bool first;
    int count, i;
    S d, di, dr, eps, f2, f2s, g2, g2s, safmin;
    S safmn2, safmx2, scale;
    T ff, fs, gs;

    // safmin = dlamch( 's' )
    cmach = 'S';
    call_lamch(cmach, safmin);

    // eps = dlamch( 'e' )
    cmach = 'E';
    call_lamch(cmach, eps);

    // safmn2 = dlamch( 'b' )**int( log( safmin / eps ) / log( dlamch( 'b' ) ) / two )
    cmach = 'B';
    S radix = 2;
    call_lamch(cmach, radix);

    int const npow = (std::log(safmin / eps) / std::log(radix) / two);
    safmn2 = std::pow(radix, npow);
    safmx2 = one / safmn2;
    scale = std::max(abs1(f), abs1(g));
    fs = f;
    gs = g;
    count = 0;

    if(scale >= safmx2)
    {
        do
        {
            count = count + 1;
            fs = fs * safmn2;
            gs = gs * safmn2;
            scale = scale * safmn2;
            has_work = ((scale >= safmx2) && (count < 20));
        } while(has_work);
    }
    else
    {
        if(scale <= safmn2)
        {
            if((g == czero) || disnan(std::abs(g)))
            {
                cs = one;
                sn = czero;
                r = f;
                return;
            }
            do
            {
                count = count - 1;
                fs = fs * safmx2;
                gs = gs * safmx2;
                scale = scale * safmx2;
                has_work = (scale <= safmn2);
            } while(has_work);
        }
        f2 = abssq(fs);
        g2 = abssq(gs);
        if(f2 <= std::max(g2, one) * safmin)
        {
            //
            //        this is a rare case: f is very small.
            //
            if(f == czero)
            {
                cs = zero;
                r = dlapy2(dble(g), dimag(g));
                //           do complex/real division explicitly with two real divisions
                d = dlapy2(dble(gs), dimag(gs));
                sn = dcmplx(dble(gs) / d, -dimag(gs) / d);
                return;
            }
            f2s = dlapy2(dble(fs), dimag(fs));
            //        g2 and g2s are accurate
            //        g2 is at least safmin, and g2s is at least safmn2
            g2s = std::sqrt(g2);
            //        error in cs from underflow in f2s is at most
            //        unfl / safmn2  <  sqrt(unfl*eps) .lt. eps
            //        if max(g2,one)=g2,  then f2  <  g2*safmin,
            //        and so cs  <  sqrt(safmin)
            //        if max(g2,one)=one,  then f2  <  safmin
            //        and so cs  <  sqrt(safmin)/safmn2 = sqrt(eps)
            //        therefore, cs = f2s/g2s / sqrt( 1 + (f2s/g2s)**2 ) = f2s/g2s
            cs = f2s / g2s;
            //        make sure abs(ff) = 1
            //        do complex/real division explicitly with 2 real divisions
            if(abs1(f) > one)
            {
                d = dlapy2(dble(f), dimag(f));
                ff = dcmplx(dble(f) / d, dimag(f) / d);
            }
            else
            {
                dr = safmx2 * dble(f);
                di = safmx2 * dimag(f);
                d = dlapy2(dr, di);
                ff = dcmplx(dr / d, di / d);
            }
            sn = ff * dcmplx(dble(gs) / g2s, -dimag(gs) / g2s);
            r = cs * f + sn * g;
        }
        else
        {
            //
            //        this is the most common case.
            //        neither f2 nor f2/g2 are less than safmin
            //        f2s cannot overflow, and it is accurate
            //
            f2s = std::sqrt(one + g2 / f2);
            //        do the f2s(real)*fs(complex) multiply with two real multiplies
            r = dcmplx(f2s * dble(fs), f2s * dimag(fs));
            cs = one / f2s;
            d = f2 + g2;
            //        do complex/real division explicitly with two real divisions
            sn = dcmplx(dble(r) / d, dimag(r) / d);
            sn = sn * dconjg(gs);
            if(count != 0)
            {
                if(count > 0)
                {
                    for(i = 1; i <= count; i++)
                    {
                        r = r * safmx2;
                    };
                }
                else
                {
                    for(i = 1; i <= -count; i++)
                    {
                        r = r * safmn2;
                    }
                }
            }
        }
    }
}

template <typename T, typename S, typename I>
static void call_scal(I& n, S& a, T& x_in, I& incx)
{
    bool const is_zero = (a == 0);
    T* const x = &x_in;
    for(I i = 0; i < n; i++)
    {
        auto const ip = i * incx;
        x[ip] *= a;
    }
}

template <typename T, typename S, typename I>
static void call_rot(I& n, T& x_in, I& incx, T& y_in, I& incy, S& c, S& s)
{
    T* const x = &(x_in);
    T* const y = &(y_in);

    for(I i = 0; i < n; i++)
    {
        auto const ix = i * incx;
        auto const iy = i * incy;

        auto const temp = c * x[ix] + s * y[iy];
        y[iy] = c * y[iy] - s * x[ix];
        x[ix] = temp;
    }
}

// --------------------------------------------------------
// lasv2 computes the singular value decomposition of a 2 x 2
// triangular matrix
// [ F G ]
// [ 0 H ]
//
// on return,
// abs(ssmax) is the larger singular value,
// abs(ssmin) is the smaller singular value,
// (csl,snl) and (csr,snr) are the left and right
// singular vectors for abs(ssmax)
//
// [ csl  snl]  [  F  G ]  [ csr   -snr] = [ ssmax   0    ]
// [-snl  csl]  [  0  H ]  [ snr    csr]   [  0     ssmin ]
// --------------------------------------------------------
template <typename T>
static void call_lasv2(T& f, T& g, T& h, T& ssmin, T& ssmax, T& snr, T& csr, T& snl, T& csl)
{
    T const zero = 0;
    T const one = 1;
    T const two = 2;
    T const four = 4;
    T const half = one / two;

    bool gasmal;
    bool swap;
    int pmax;
    char cmach;

    T a, clt, crt, d, fa, ft, ga, gt, ha, ht, l, m;
    T mm, r, s, slt, srt, t, temp, tsign, tt;
    T macheps;

    auto sign = [](auto a, auto b) {
        auto const abs_a = std::abs(a);
        return ((b >= 0) ? abs_a : -abs_a);
    };

    ft = f;
    fa = std::abs(ft);
    ht = h;
    ha = std::abs(h);
    //
    //     pmax points to the maximum absolute element of matrix
    //       pmax = 1 if f largest in absolute values
    //       pmax = 2 if g largest in absolute values
    //       pmax = 3 if h largest in absolute values
    //
    pmax = 1;
    swap = (ha > fa);
    if(swap)
    {
        pmax = 3;
        temp = ft;
        ft = ht;
        ht = temp;
        temp = fa;
        fa = ha;
        ha = temp;
        //
        //        now fa >= ha
        //
    }
    gt = g;
    ga = std::abs(gt);
    if(ga == zero)
    {
        //
        //        diagonal matrix
        //
        ssmin = ha;
        ssmax = fa;
        clt = one;
        crt = one;
        slt = zero;
        srt = zero;
    }
    else
    {
        gasmal = true;
        if(ga > fa)
        {
            pmax = 2;

            cmach = 'E';
            call_lamch(cmach, macheps);

            if((fa / ga) < macheps)
            {
                //
                //              case of very large ga
                //
                gasmal = false;
                ssmax = ga;
                if(ha > one)
                {
                    ssmin = fa / (ga / ha);
                }
                else
                {
                    ssmin = (fa / ga) * ha;
                }
                clt = one;
                slt = ht / gt;
                srt = one;
                crt = ft / gt;
            }
        }
        if(gasmal)
        {
            //
            //           normal case
            //
            d = fa - ha;
            if(d == fa)
            {
                //
                //              copes with infinite f or h
                //
                l = one;
            }
            else
            {
                l = d / fa;
            }
            //
            //           note that 0  <=  l <= 1
            //
            m = gt / ft;
            //
            //           note that abs(m)  <=  1/macheps
            //
            t = two - l;
            //
            //           note that t >= 1
            //
            mm = m * m;
            tt = t * t;
            s = std::sqrt(tt + mm);
            //
            //           note that 1  <=  s <= 1 + 1/macheps
            //
            if(l == zero)
            {
                r = std::abs(m);
            }
            else
            {
                r = std::sqrt(l * l + mm);
            }
            //
            //           note that 0  <=  r .le. 1 + 1/macheps
            //
            a = half * (s + r);
            //
            //           note that 1  <=  a .le. 1 + abs(m)
            //
            ssmin = ha / a;
            ssmax = fa * a;
            if(mm == zero)
            {
                //
                //              note that m is very tiny
                //
                if(l == zero)
                {
                    t = sign(two, ft) * sign(one, gt);
                }
                else
                {
                    t = gt / sign(d, ft) + m / t;
                }
            }
            else
            {
                t = (m / (s + t) + m / (r + l)) * (one + a);
            }
            l = std::sqrt(t * t + four);
            crt = two / l;
            srt = t / l;
            clt = (crt + srt * m) / a;
            slt = (ht / ft) * srt / a;
        }
    }
    if(swap)
    {
        csl = srt;
        snl = crt;
        csr = slt;
        snr = clt;
    }
    else
    {
        csl = clt;
        snl = slt;
        csr = crt;
        snr = srt;
    }
    //
    //     correct signs of ssmax and ssmin
    //
    if(pmax == 1)
    {
        tsign = sign(one, csr) * sign(one, csl) * sign(one, f);
    }
    if(pmax == 2)
    {
        tsign = sign(one, snr) * sign(one, csl) * sign(one, g);
    }
    if(pmax == 3)
    {
        tsign = sign(one, snr) * sign(one, snl) * sign(one, h);
    }
    ssmax = sign(ssmax, tsign);
    ssmin = sign(ssmin, tsign * sign(one, f) * sign(one, h));
}

ROCSOLVER_END_NAMESPACE

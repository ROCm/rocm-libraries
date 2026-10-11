/* **************************************************************************
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

#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "common/misc/clientcommon.hpp"

/* Helpers shared by the tests of the equilibration routines (GEEQU, GEEQUB, POEQU, POEQUB, LAQGE,
   LAQSY and LAQHE).

   The test matrices do not come from rocblas_init: its small integers include exact powers of 2, for
   which the power-of-2 rounding INT(LOG(x)/LOG(2)) of GEEQUB and POEQUB depends on how the logarithm
   is rounded. Instead, every nonzero magnitude is u * 2^e with u in [0.55, 0.95], so that the
   fractional part of its base-2 logarithm (and of half of it) stays at least 0.03 away from an
   integer. Rows, columns and diagonal elements are scaled by factors 2^e spanning many orders of
   magnitude, so that the computed scalings are not trivial. */

// exponent range of the row and column factors (2^20 is about 10^6)
constexpr int equ_spread = 20;

inline int equ_rand_int(int lo, int hi)
{
    return std::uniform_int_distribution<int>(lo, hi)(rocblas_rng);
}

inline double equ_rand_sign()
{
    return equ_rand_int(0, 1) ? 1.0 : -1.0;
}

// when set, the generated magnitudes are exact powers of 2 (test case 'P')
inline bool equ_exact_powers = false;

// random positive value u * 2^e, with u in [0.55, 0.95] (or u = 1 if equ_exact_powers is set)
template <typename S>
S equ_positive(int e)
{
    if(equ_exact_powers)
        return std::ldexp(S(1), e);
    S u = S(std::uniform_real_distribution<double>(0.55, 0.95)(rocblas_rng));
    return std::ldexp(u, e);
}

// builds a scalar from its real and imaginary parts (the latter is ignored for real types)
template <typename T, typename S>
T equ_make(S re, S im)
{
    if constexpr(rocblas_is_complex<T>)
        return T(re, im);
    else
        return T(re);
}

/* Random element with random sign(s) and absolute value u * 2^e. For complex types, the absolute
   value |Re(a)| + |Im(a)| used by LAPACK is exactly u * 2^e: it is split between the two parts as
   t = p + q with t/2 <= p <= t, so that q = t - p is exact. */
template <typename T>
T equ_element(int e)
{
    using S = decltype(std::real(T{}));
    S t = equ_positive<S>(e);
    if constexpr(rocblas_is_complex<T>)
    {
        S p = equ_exact_powers
            ? t * S(0.75)
            : t * S(std::uniform_real_distribution<double>(0.5, 0.8)(rocblas_rng));
        S q = t - p;
        if(equ_rand_int(0, 1))
            std::swap(p, q);
        return T(S(equ_rand_sign()) * p, S(equ_rand_sign()) * q);
    }
    else
        return T(S(equ_rand_sign()) * t);
}

/* Exponent offset of the generated values, whose exponents (before the offset) lie in
   [-range, range]: 'N' and 'P' keep the magnitudes around 1, 'L' moves the largest magnitude close
   to 2^(max_exponent - 8) (near overflow), 'S' moves the smallest magnitude close to
   2^(min_exponent + 8) (near underflow), and 'D' makes the largest magnitudes subnormal (some
   elements may underflow to zero). */
template <typename S>
int equ_shift(const char scale, const int range)
{
    if(scale == 'D')
        return std::numeric_limits<S>::min_exponent - 2 - range;
    if(scale == 'L')
        return std::numeric_limits<S>::max_exponent - 8 - range;
    if(scale == 'S')
        return std::numeric_limits<S>::min_exponent + 9 + range;
    return 0;
}

// large finite value, used for entries that must not be read
template <typename S>
S equ_huge()
{
    return std::ldexp(S(0.75), std::numeric_limits<S>::max_exponent - 2);
}

// NaN with random payload, used for entries that must not be read nor modified
template <typename T>
T equ_nan()
{
    return T(rocblas_nan_rng());
}

/* LAPACK's thresholds for LAQGE, LAQSY and LAQHE: amax must lie in [small, large] for the scaling
   to be skipped, with small = safe minimum / precision. */
template <typename S>
S equ_small()
{
    return S(get_safemin<S>() / get_epsilon<S>());
}

template <typename S>
S equ_large()
{
    return S(1) / equ_small<S>();
}

/* Fills the m-by-n general matrix A with a(i,j) = u * 2^(c_j - r_i + shift), with random r_i and c_j
   in [0, equ_spread] and shift given by scale (see equ_shift). Row zero_row and column zero_col
   (1-based; 0 for none) are set to zero. The padding rows m to lda-1 are filled with NaN if
   nan_padding is true, or with large values otherwise. */
template <typename T, typename I>
void equ_init_general(T* A,
                      const I m,
                      const I n,
                      const I lda,
                      const char scale,
                      const I zero_row,
                      const I zero_col,
                      const bool nan_padding)
{
    using S = decltype(std::real(T{}));
    const int shift = equ_shift<S>(scale, equ_spread);
    equ_exact_powers = (scale == 'P');

    std::vector<int> r(m), c(n);
    for(auto& e : r)
        e = equ_rand_int(0, equ_spread);
    for(auto& e : c)
        e = equ_rand_int(0, equ_spread);

    for(I j = 0; j < n; j++)
    {
        for(I i = 0; i < m; i++)
        {
            if(i + 1 == zero_row || j + 1 == zero_col)
                A[i + j * lda] = T(S(0));
            else
                A[i + j * lda] = equ_element<T>(c[j] - r[i] + shift);
        }
        for(I i = m; i < lda; i++)
            A[i + j * lda] = nan_padding ? equ_nan<T>() : equ_make<T>(equ_huge<S>(), equ_huge<S>());
    }
    equ_exact_powers = false;
}

/* Fills the n-by-n matrix A with a positive diagonal of magnitudes u * 2^e, with e in
   [-2*equ_spread, 2*equ_spread] plus the shift given by scale (see equ_shift). Everything that
   POEQU/POEQUB must not read (off-diagonal entries, padding rows and, for complex types, the
   imaginary part of the diagonal) holds NaN or large values. The diagonal elements zero_diag and
   neg_diag (1-based; 0 for none) are set to zero and to a negative value, respectively. */
template <typename T, typename I>
void equ_init_diagonal(T* A, const I n, const I lda, const char scale, const I zero_diag, const I neg_diag)
{
    using S = decltype(std::real(T{}));
    const int shift = equ_shift<S>(scale, 2 * equ_spread);
    const S big = equ_huge<S>();
    equ_exact_powers = (scale == 'P');

    for(I j = 0; j < n; j++)
    {
        for(I i = 0; i < lda; i++)
        {
            if(i == j)
            {
                S d = equ_positive<S>(equ_rand_int(-2 * equ_spread, 2 * equ_spread) + shift);
                if(i + 1 == zero_diag)
                    d = 0;
                else if(i + 1 == neg_diag)
                    d = -d;
                A[i + j * lda] = equ_make<T>(d, big);
            }
            else if((i + j) % 2)
                A[i + j * lda] = equ_make<T>(-big, -big);
            else
                A[i + j * lda] = equ_nan<T>();
        }
    }
    equ_exact_powers = false;
}

/* Fills the triangle uplo of the n-by-n matrix A with random entries: the diagonal has a positive real
   part u * 2^e with e in [-2*equ_spread, 2*equ_spread] (and, for complex types, a nonzero imaginary
   part), and the off-diagonal entries have exponents in [-equ_spread/2, equ_spread/2]. The other
   triangle and the padding rows hold NaN; they must not be referenced. */
template <typename T, typename I>
void equ_init_triangle(T* A, const rocblas_fill uplo, const I n, const I lda)
{
    using S = decltype(std::real(T{}));

    for(I j = 0; j < n; j++)
    {
        for(I i = 0; i < lda; i++)
        {
            bool referenced = i < n && (uplo == rocblas_fill_upper ? i <= j : i >= j);
            if(!referenced)
                A[i + j * lda] = equ_nan<T>();
            else if(i == j)
                A[i + j * lda] = equ_make<T>(
                    equ_positive<S>(equ_rand_int(-2 * equ_spread, 2 * equ_spread)),
                    S(equ_rand_sign()) * equ_positive<S>(equ_rand_int(-equ_spread, equ_spread)));
            else
                A[i + j * lda] = equ_element<T>(equ_rand_int(-equ_spread / 2, equ_spread / 2));
        }
    }
}

template <typename T>
bool equ_same_bits(const T& a, const T& b)
{
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

/* Relative error of res with respect to ref. Equal values (including equal sentinels and zeros) give
   zero; any other difference with respect to a zero or non-finite value is infinite. */
template <typename S>
double equ_relerr(const S res, const S ref)
{
    if(res == ref)
        return 0;
    if(ref == 0 || !std::isfinite(ref) || !std::isfinite(res))
        return std::numeric_limits<double>::infinity();
    return std::abs(double(res) - double(ref)) / std::abs(double(ref));
}

// same, part by part for complex types
template <typename T>
double equ_relerr_elem(const T res, const T ref)
{
    if constexpr(rocblas_is_complex<T>)
        return std::max(equ_relerr(res.real(), ref.real()), equ_relerr(res.imag(), ref.imag()));
    else
        return equ_relerr(res, ref);
}

/* Largest relative error, element by element, of the result matrix with respect to the reference
   matrix (both of size lda-by-n). Entries for which referenced(i, j) is false must be bitwise
   identical (they are not referenced by the routine); otherwise the error is infinite. */
template <typename T, typename I, typename F>
double equ_matrix_error(const T* res, const T* ref, const I n, const I lda, F referenced)
{
    double err = 0;
    for(I j = 0; j < n; j++)
    {
        for(I i = 0; i < lda; i++)
        {
            const T& x = res[i + j * lda];
            const T& y = ref[i + j * lda];
            if(equ_same_bits(x, y))
                continue;
            err = std::max(err,
                           referenced(i, j) ? equ_relerr_elem(x, y)
                                            : std::numeric_limits<double>::infinity());
        }
    }
    return err;
}

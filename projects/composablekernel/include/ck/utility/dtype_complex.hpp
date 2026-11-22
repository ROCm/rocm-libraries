// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/type.hpp"

namespace ck {

// A complex number with float or double parts. The real part comes first, there is no padding,
// and the alignment is twice the size of a part: the same size, order and alignment as
// hipFloatComplex / hipDoubleComplex, cuda::std::complex and thrust::complex. std::complex<T> is
// aligned to T only, so an array of std::complex is copied to complex_t, not cast.
// Kernels count one complex_t as one element and load or store it as two components (see
// is_two_component_type in data_type.hpp).
template <typename T>
struct alignas(2 * sizeof(T)) complex_t
{
    T real;
    T imag;

    static_assert(is_same_v<T, float> || is_same_v<T, double>,
                  "complex_t supports float and double components");

    // Marks the type as a two-component element.
    using component_type = T;
    // Read by non_native_vector_base for 16-byte elements.
    using element_type = T;

    // Trivial, so that kernels can keep complex_t in __shared__ arrays.
    complex_t() = default;
    // Explicit, so that a real number never becomes a complex one by accident.
    __host__ __device__ constexpr explicit complex_t(T r) : real{r}, imag{0} {}
    __host__ __device__ constexpr complex_t(T r, T i) : real{r}, imag{i} {}
};

using complex_float_t  = complex_t<float>;
using complex_double_t = complex_t<double>;

static_assert(sizeof(complex_float_t) == 2 * sizeof(float), "complex_float_t layout");
static_assert(sizeof(complex_double_t) == 2 * sizeof(double), "complex_double_t layout");
static_assert(alignof(complex_float_t) == 2 * sizeof(float) &&
                  alignof(complex_double_t) == 2 * sizeof(double),
              "complex_t must be aligned to twice the size of a part");
static_assert(is_trivially_copyable_v<complex_float_t> && is_trivially_copyable_v<complex_double_t>,
              "complex_t must be trivially copyable");
static_assert(__is_standard_layout(complex_float_t) && __is_standard_layout(complex_double_t),
              "complex_t must be standard layout");
static_assert(__is_trivially_constructible(complex_float_t) &&
                  __is_trivially_constructible(complex_double_t),
              "complex_t must be trivially default constructible");

namespace complex_impl {

__host__ __device__ inline float complex_fma(float a, float b, float c)
{
    return __builtin_fmaf(a, b, c);
}

__host__ __device__ inline double complex_fma(double a, double b, double c)
{
    return __builtin_fma(a, b, c);
}

// a * d - b * c with relative error at most 2u, u the unit roundoff, also under cancellation,
// when nothing underflows (Kahan's algorithm; the bound is from Jeannerod, Louvet and Muller,
// Math. Comp. 82, 2013): the rounding error of b * c is computed with a fused multiply-add and
// added back. The error is taken as b * c - w and subtracted, so that an exact zero result
// keeps the sign IEEE arithmetic gives a * d - b * c.
template <typename T>
__host__ __device__ inline T diff_of_products(T a, T d, T b, T c)
{
    const T w = b * c;
    const T e = complex_fma(b, c, -w);
    const T f = complex_fma(a, d, -w);
    return f - e;
}

__host__ __device__ inline float complex_copysign(float x, float s)
{
    return __builtin_copysignf(x, s);
}

__host__ __device__ inline double complex_copysign(double x, double s)
{
    return __builtin_copysign(x, s);
}

__host__ __device__ inline float complex_ldexp(float x, int n) { return __builtin_ldexpf(x, n); }

__host__ __device__ inline double complex_ldexp(double x, int n) { return __builtin_ldexp(x, n); }

// The exponent of x, as ilogb gives it, for finite nonzero x. Through frexp, which the GPU
// has as an instruction; ilogb is a library call there.
__host__ __device__ inline int complex_ilogb(float x)
{
    int e = 0;
    __builtin_frexpf(x, &e);
    return e - 1;
}

__host__ __device__ inline int complex_ilogb(double x)
{
    int e = 0;
    __builtin_frexp(x, &e);
    return e - 1;
}

__host__ __device__ inline float complex_fabs(float x) { return __builtin_fabsf(x); }

__host__ __device__ inline double complex_fabs(double x) { return __builtin_fabs(x); }

template <typename T>
__host__ __device__ inline complex_t<T> kahan_multiply(const complex_t<T>& a, const complex_t<T>& b)
{
    return complex_t<T>{diff_of_products(a.real, b.real, a.imag, b.imag),
                        diff_of_products(a.real, b.imag, -a.imag, b.real)};
}

__host__ __device__ constexpr int complex_max_exponent(float) { return __FLT_MAX_EXP__ - 1; }

__host__ __device__ constexpr int complex_max_exponent(double) { return __DBL_MAX_EXP__ - 1; }

// a * d - b * c for finite a, b, c, d when a product overflows. Both products are scaled by the
// same power of two, through their first factors a and b, so that the larger product lies just
// below the overflow threshold. Kahan's algorithm then runs without overflow, and scaling the
// result back is exact unless the result itself is too large for T, which gives an infinity with
// the correct sign. A zero product is left out of the scale. The scaled factor of the larger
// product stays normal; that of the other turns subnormal only when its product is more than
// 2^124 (float) or 2^1020 (double) below the larger one, where it changes no rounding.
template <typename T>
__host__ __device__ inline T rescaled_diff_of_products(T a, T d, T b, T c)
{
    const bool ad_nonzero = complex_fabs(a) > T{0} && complex_fabs(d) > T{0};
    const bool bc_nonzero = complex_fabs(b) > T{0} && complex_fabs(c) > T{0};
    const int e_ad        = ad_nonzero ? complex_ilogb(a) + complex_ilogb(d) : 0;
    const int e_bc        = bc_nonzero ? complex_ilogb(b) + complex_ilogb(c) : 0;
    const int e_max       = !bc_nonzero || (ad_nonzero && e_ad > e_bc) ? e_ad : e_bc;
    const int shift       = e_max - (complex_max_exponent(T{}) - 2);
    if(shift <= 0)
    {
        return diff_of_products(a, d, b, c);
    }
    return complex_ldexp(diff_of_products(complex_ldexp(a, -shift), d, complex_ldexp(b, -shift), c),
                         shift);
}

// Finite factors whose product came out infinite or NaN: a product overflowed. Each part that is
// not finite is computed again with the products rescaled.
template <typename T>
__host__ __device__ inline complex_t<T> rescaled_multiply(const complex_t<T>& a,
                                                          const complex_t<T>& b)
{
    complex_t<T> r = kahan_multiply(a, b);
    if(!__builtin_isfinite(r.real))
    {
        r.real = rescaled_diff_of_products(a.real, b.real, a.imag, b.imag);
    }
    if(!__builtin_isfinite(r.imag))
    {
        r.imag = rescaled_diff_of_products(a.real, b.imag, -a.imag, b.real);
    }
    return r;
}

// An infinity keeps its sign as 1, any finite part becomes a zero of its sign.
template <typename T>
__host__ __device__ inline T box_infinity(T x)
{
    return complex_copysign(__builtin_isinf(x) ? T{1} : T{0}, x);
}

template <typename T>
__host__ __device__ inline T nan_to_zero(T x)
{
    return __builtin_isnan(x) ? complex_copysign(T{0}, x) : x;
}

// A factor is infinite or NaN: the products of the textbook formula, then the recovery of
// C11 Annex G.5.1 (its example function _Cmultd). A result of NaN + iNaN is replaced by an
// infinity when a factor is infinite, or when a product overflowed, because in Annex G an
// infinity times a nonzero value is an infinity.
// C11, final public draft N1570: https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
template <typename T>
__host__ __device__ inline complex_t<T> annex_g_multiply(const complex_t<T>& z,
                                                         const complex_t<T>& u)
{
#pragma clang fp contract(off)
    T a = z.real, b = z.imag, c = u.real, d = u.imag;
    const T ac = a * c, bd = b * d, ad = a * d, bc = b * c;
    T x = ac - bd, y = ad + bc;
    if(__builtin_isnan(x) && __builtin_isnan(y))
    {
        bool recalc = false;
        if(__builtin_isinf(a) || __builtin_isinf(b))
        {
            a      = box_infinity(a);
            b      = box_infinity(b);
            c      = nan_to_zero(c);
            d      = nan_to_zero(d);
            recalc = true;
        }
        if(__builtin_isinf(c) || __builtin_isinf(d))
        {
            c      = box_infinity(c);
            d      = box_infinity(d);
            a      = nan_to_zero(a);
            b      = nan_to_zero(b);
            recalc = true;
        }
        if(!recalc && (__builtin_isinf(ac) || __builtin_isinf(bd) || __builtin_isinf(ad) ||
                       __builtin_isinf(bc)))
        {
            a      = nan_to_zero(a);
            b      = nan_to_zero(b);
            c      = nan_to_zero(c);
            d      = nan_to_zero(d);
            recalc = true;
        }
        if(recalc)
        {
            const T inf = static_cast<T>(__builtin_huge_valf());
            x           = inf * (a * c - b * d);
            y           = inf * (a * d + b * c);
        }
    }
    return complex_t<T>{x, y};
}

// The product when Kahan's result is not finite: finite factors are rescaled, others follow
// Annex G. Not inlined, so that the rare slow path is not copied into every multiply.
template <typename T>
__host__ __device__ __attribute__((noinline)) complex_t<T>
multiply_non_finite(const complex_t<T>& a, const complex_t<T>& b)
{
    if(__builtin_isfinite(a.real) && __builtin_isfinite(a.imag) && __builtin_isfinite(b.real) &&
       __builtin_isfinite(b.imag))
    {
        return rescaled_multiply(a, b);
    }
    return annex_g_multiply(a, b);
}

} // namespace complex_impl

template <typename T>
__host__ __device__ constexpr complex_t<T> operator+(const complex_t<T>& a, const complex_t<T>& b)
{
    return complex_t<T>{a.real + b.real, a.imag + b.imag};
}

template <typename T>
__host__ __device__ constexpr complex_t<T> operator-(const complex_t<T>& a, const complex_t<T>& b)
{
    return complex_t<T>{a.real - b.real, a.imag - b.imag};
}

template <typename T>
__host__ __device__ constexpr complex_t<T> operator-(const complex_t<T>& a)
{
    return complex_t<T>{-a.real, -a.imag};
}

// Each component is a difference of two products. Computed with Kahan's algorithm so that
// cancellation does not lose the result and every target rounds the same way. A result that is
// not finite is computed again as C11 Annex G requires: finite factors give an infinity only when
// the result is too large for T, and infinite factors give an infinity, not NaN + iNaN.
// C11, final public draft N1570: https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf
template <typename T>
__host__ __device__ inline complex_t<T> operator*(const complex_t<T>& a, const complex_t<T>& b)
{
    const complex_t<T> r = complex_impl::kahan_multiply(a, b);
    // One test for both parts: the sum is not finite when either part is not. A finite sum that
    // overflows takes the slow path too, which then returns the same result.
    if(__builtin_isfinite(r.real + r.imag))
    {
        return r;
    }
    return complex_impl::multiply_non_finite(a, b);
}

template <typename T>
__host__ __device__ constexpr complex_t<T>& operator+=(complex_t<T>& a, const complex_t<T>& b)
{
    a = a + b;
    return a;
}

template <typename T>
__host__ __device__ constexpr complex_t<T>& operator-=(complex_t<T>& a, const complex_t<T>& b)
{
    a = a - b;
    return a;
}

template <typename T>
__host__ __device__ inline complex_t<T>& operator*=(complex_t<T>& a, const complex_t<T>& b)
{
    a = a * b;
    return a;
}

// Written with <= and >= because CK builds with -Wfloat-equal; same result as ==, including NaN.
template <typename T>
__host__ __device__ constexpr bool operator==(const complex_t<T>& a, const complex_t<T>& b)
{
    return a.real <= b.real && a.real >= b.real && a.imag <= b.imag && a.imag >= b.imag;
}

template <typename T>
__host__ __device__ constexpr bool operator!=(const complex_t<T>& a, const complex_t<T>& b)
{
    return !(a == b);
}

template <typename T>
__host__ __device__ constexpr complex_t<T> conj(const complex_t<T>& a)
{
    return complex_t<T>{a.real, -a.imag};
}

} // namespace ck

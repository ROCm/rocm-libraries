// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/type.hpp"

namespace ck {

// A complex number with float or double parts. The real part comes first, there is no padding,
// and the alignment is twice the size of a part: the same size, order and alignment as
// hipFloatComplex / hipDoubleComplex, cuda::std::complex and thrust::complex. std::complex<T> is
// aligned to T only, so an array of std::complex is copied to complex_t, not cast.
template <typename T>
struct alignas(2 * sizeof(T)) complex_t
{
    T real;
    T imag;

    static_assert(is_same_v<T, float> || is_same_v<T, double>,
                  "complex_t supports float and double components");

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
// cancellation does not lose the result and every target rounds the same way.
template <typename T>
__host__ __device__ inline complex_t<T> operator*(const complex_t<T>& a, const complex_t<T>& b)
{
    return complex_t<T>{complex_impl::diff_of_products(a.real, b.real, a.imag, b.imag),
                        complex_impl::diff_of_products(a.real, b.imag, -a.imag, b.real)};
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

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

} // namespace ck

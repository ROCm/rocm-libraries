//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// UNSUPPORTED: hipcc
// <cuda/std/complex>

// template<Arithmetic T>
//   T
//   imag(const T& x);

#include <cuda/std/cassert>
#include <cuda/std/complex>
#include <cuda/std/type_traits>

#include "../cases.h"
#include "test_macros.h"

template <class T, int x, class Target>
__host__ __device__ void test_nonconstexpr()
{
  static_assert((cuda::std::is_same<decltype(cuda::std::imag(T(x))), Target>::value), "");
  assert(cuda::std::imag(T(x)) == T(0));
}

template <class T, int x>
__host__ __device__ void test(typename cuda::std::enable_if<cuda::std::is_integral<T>::value>::type* = 0)
{
  test_nonconstexpr<T, x, double>();

  static_assert((cuda::std::is_same<decltype(cuda::std::imag(T(x))), double>::value), "");
  assert(cuda::std::imag(x) == 0);
#if TEST_STD_VER > 2011
  constexpr T val{x};
  static_assert(cuda::std::imag(val) == T(0), "");
  constexpr cuda::std::complex<T> t{val, val};
  static_assert(t.imag() == T(x), "");
#endif
}

template <class T, int x>
__host__ __device__ void test(typename cuda::std::enable_if<!cuda::std::is_integral<T>::value>::type* = 0)
{
  test_nonconstexpr<T, x, T>();

  static_assert((cuda::std::is_same<decltype(cuda::std::imag(T(x))), T>::value), "");
  assert(cuda::std::imag(x) == 0);
#if TEST_STD_VER > 2011
  constexpr T val{x};
  static_assert(cuda::std::imag(val) == T(0), "");
  constexpr cuda::std::complex<T> t{val, val};
  static_assert(t.imag() == T(x), "");
#endif
}

template <class T>
__host__ __device__ void test_nonconstexpr()
{
  test_nonconstexpr<T, 0, T>();
  test_nonconstexpr<T, 1, T>();
  test_nonconstexpr<T, 10, T>();
}

template <class T>
__host__ __device__ void test()
{
  test<T, 0>();
  test<T, 1>();
  test<T, 10>();
}

int main(int, char**)
{
  test<float>();
  test<double>();
// CUDA treats long double as double
//  test<long double>();
#ifdef _LIBCUDACXX_HAS_NVFP16
  test_nonconstexpr<__half>();
#endif
#ifdef _LIBCUDACXX_HAS_NVBF16
  test_nonconstexpr<__nv_bfloat16>();
#endif
  test<int>();
  test<unsigned>();
  test<long long>();

  return 0;
}

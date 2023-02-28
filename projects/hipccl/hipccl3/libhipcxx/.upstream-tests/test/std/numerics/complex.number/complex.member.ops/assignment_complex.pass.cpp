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

// complex& operator=(const complex&);
// template<class X> complex& operator= (const complex<X>&);

#if defined(_MSC_VER)
#  pragma warning(disable : 4244) // conversion from 'const double' to 'int', possible loss of data
#endif

#include <cuda/std/cassert>
#include <cuda/std/complex>

#include "test_macros.h"

template <class T, class X>
__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  cuda::std::complex<T> c;
  assert(c.real() == T(0));
  assert(c.imag() == T(0));
  cuda::std::complex<T> c2(1.5, 2.5);
  c = c2;
  assert(c.real() == T(1.5));
  assert(c.imag() == T(2.5));
  cuda::std::complex<X> c3(3.5, -4.5);
  c = c3;
  assert(c.real() == T(3.5));
  assert(c.imag() == T(-4.5));

  return true;
}

int main(int, char**)
{
  test<float, float>();
  test<float, double>();

  test<double, float>();
  test<double, double>();

  // CUDA treats long double as double
  //  test<float, long double>();
  //  test<double, long double>();
  //  test<long double, float>();
  //  test<long double, double>();
  //  test<long double, long double>();

#ifdef _LIBCUDACXX_HAS_NVFP16
  test<float, __half>();
  test<double, __half>();
  test<__half, float>();
  test<__half, double>();
#  ifdef _LIBCUDACXX_HAS_NVBF16
  test<float, __nv_bfloat16>();
  test<double, __nv_bfloat16>();
  test<__nv_bfloat16, float>();
  test<__nv_bfloat16, double>();
#  endif
#endif

#if TEST_STD_VER > 2011
  static_assert(test<float, float>(), "");
  static_assert(test<float, double>(), "");

  static_assert(test<double, float>(), "");
  static_assert(test<double, double>(), "");
#endif // TEST_STD_VER > 2011

  return 0;
}

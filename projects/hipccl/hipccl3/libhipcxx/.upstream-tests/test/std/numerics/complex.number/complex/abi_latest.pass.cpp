//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// MIT License
//
// Modifications Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// HIP does not support long double, but it silently treats it as double without emitting a
// compiler error. Therefore, this test is unsupported for HIP. Please see "Use of Long
// Double Type" at https://docs.amd.com/bundle/HIP-Programming-Guide-v5.1/page/Programming_with_HIP.html.
// UNSUPPORTED: hipcc, hiprtc
// <cuda/std/complex>

// template<class T>
// class complex
// {
// public:
//   typedef T value_type;
//   ...
// };

#include <cuda/std/complex>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test()
{
  typedef cuda::std::complex<T> C;

  static_assert(sizeof(C) == (sizeof(T) * 2), "wrong size");
  static_assert(alignof(C) == (alignof(T) * 2), "misaligned");
}

int main(int, char**)
{
  test<float>();
  test<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _LIBCUDACXX_HAS_NVFP16()
  test<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

// test numeric_limits

// lowest()

#include <cuda/std/cassert>
#include <cuda/std/cfloat>
#include <cuda/std/climits>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include "common.h"
#include "test_macros.h"

template <class T>
__host__ __device__ void test(T expected)
{
  assert(float_eq(cuda::std::numeric_limits<T>::lowest(), expected));
  assert(cuda::std::numeric_limits<T>::is_bounded);
  assert(float_eq(cuda::std::numeric_limits<const T>::lowest(), expected));
  assert(cuda::std::numeric_limits<const T>::is_bounded);
  assert(float_eq(cuda::std::numeric_limits<volatile T>::lowest(), expected));
  assert(cuda::std::numeric_limits<volatile T>::is_bounded);
  assert(float_eq(cuda::std::numeric_limits<const volatile T>::lowest(), expected));
  assert(cuda::std::numeric_limits<const volatile T>::is_bounded);
}

int main(int, char**)
{
  test<bool>(false);
  test<char>(CHAR_MIN);

  test<signed char>(SCHAR_MIN);
  test<unsigned char>(0);
#if !defined(TEST_COMPILER_NVRTC) && !defined(TEST_COMPILER_HIPRTC)
  test<wchar_t>(WCHAR_MIN);
#endif
#if TEST_STD_VER > 2017 && defined(__cpp_char8_t)
  test<char8_t>(0);
#endif
#ifndef _LIBCUDACXX_HAS_NO_UNICODE_CHARS
  test<char16_t>(0);
  test<char32_t>(0);
#endif // _LIBCUDACXX_HAS_NO_UNICODE_CHARS
  test<short>(SHRT_MIN);
  test<unsigned short>(0);
  test<int>(INT_MIN);
  test<unsigned int>(0);
  test<long>(LONG_MIN);
  test<unsigned long>(0);
  test<long long>(LLONG_MIN);
  test<unsigned long long>(0);
#ifndef _LIBCUDACXX_HAS_NO_INT128
  test<__int128_t>(-__int128_t(__uint128_t(-1) / 2) - 1);
  test<__uint128_t>(0);
#endif
  test<float>(-FLT_MAX);
  test<double>(-DBL_MAX);
#ifndef _LIBCUDACXX_HAS_NO_LONG_DOUBLE
  test<long double>(-LDBL_MAX);
#endif
#if _CCCL_HAS_NVFP16()
  test<__half>(__double2half(-65504.0));
#endif // _CCCL_HAS_NVFP16
#if _CCCL_HAS_NVBF16()
  test<__nv_bfloat16>(__double2bfloat16(-3.3895313892515355e+38));
#endif // _CCCL_HAS_NVBF16
#if _CCCL_HAS_NVFP8()
  test<__nv_fp8_e4m3>(make_fp8_e4m3(-448.0));
  test<__nv_fp8_e5m2>(make_fp8_e5m2(-57344.0));
#endif // _CCCL_HAS_NVFP8()

  return 0;
}

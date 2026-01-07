//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// MIT License
//
// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

// <cuda/std/complex>

// template<class T>
//   complex<T>
//   tanh(const complex<T>& x);

#include <cuda/std/cassert>
#include <cuda/std/complex>

#include "../cases.h"
#include "test_macros.h"

template <class T>
__host__ __device__ void test(const cuda::std::complex<T>& c, cuda::std::complex<T> x)
{
  assert(tanh(c) == x);
}

template <class T>
__host__ __device__ void test()
{
  test(cuda::std::complex<T>(0, 0), cuda::std::complex<T>(0, 0));
}

template <class T>
__host__ __device__ void test_edges()
{
  auto testcases   = get_testcases<T>();
  const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
  for (unsigned i = 0; i < N; ++i)
  {
    cuda::std::complex<T> r = cuda::std::tanh(testcases[i]);
    if (testcases[i].real() == T(0) && testcases[i].imag() == T(0))
    {
      assert(r.real() == T(0));
      assert(cuda::std::signbit(r.real()) == cuda::std::signbit(testcases[i].real()));
      assert(r.imag() == T(0));
      assert(cuda::std::signbit(r.imag()) == cuda::std::signbit(testcases[i].imag()));
    }
    else if (cuda::std::isfinite(testcases[i].real()) && cuda::std::isinf(testcases[i].imag()))
    {
      assert(cuda::std::isnan(r.real()));
      assert(cuda::std::isnan(r.imag()));
    }
    else if (cuda::std::isfinite(testcases[i].real()) && cuda::std::isnan(testcases[i].imag()))
    {
      assert(cuda::std::isnan(r.real()));
      assert(cuda::std::isnan(r.imag()));
    }
    else if (cuda::std::isinf(testcases[i].real()) && cuda::std::isfinite(testcases[i].imag()))
    {
      assert(r.real() == (testcases[i].real() > T(0) ? T(1) : -T(1)));
      assert(r.imag() == T(0));
      assert(cuda::std::signbit(r.imag()) == cuda::std::signbit(cuda::std::sin(T(2) * testcases[i].imag())));
    }
    else if (cuda::std::isinf(testcases[i].real()) && cuda::std::isinf(testcases[i].imag()))
    {
      assert(r.real() == (testcases[i].real() > T(0) ? T(1) : -T(1)));
      assert(r.imag() == T(0));
    }
    else if (cuda::std::isinf(testcases[i].real()) && cuda::std::isnan(testcases[i].imag()))
    {
      assert(r.real() == (testcases[i].real() > T(0) ? T(1) : -T(1)));
      assert(r.imag() == T(0));
    }
    else if (cuda::std::isnan(testcases[i].real()) && testcases[i].imag() == T(0))
    {
      assert(cuda::std::isnan(r.real()));
      assert(r.imag() == T(0));
      assert(cuda::std::signbit(r.imag()) == cuda::std::signbit(testcases[i].imag()));
    }
    else if (cuda::std::isnan(testcases[i].real()) && cuda::std::isfinite(testcases[i].imag()))
    {
      assert(cuda::std::isnan(r.real()));
      assert(cuda::std::isnan(r.imag()));
    }
    else if (cuda::std::isnan(testcases[i].real()) && cuda::std::isnan(testcases[i].imag()))
    {
      assert(cuda::std::isnan(r.real()));
      assert(cuda::std::isnan(r.imag()));
    }
  }
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
  test_edges<double>();
#if _LIBCUDACXX_HAS_NVFP16()
  test_edges<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
// NOTE(HIP/AMD): for specific ROCm versions the optimization causes test failures for __hip_bfloat16 (compare internal issue 134 or https://github.com/ROCm/libhipcxx/issues/13)
#if !defined(__OPTIMIZE__) || (defined(ROCM_VERSION_MAJOR) and defined(ROCM_VERSION_MINOR) and (ROCM_VERSION_MAJOR < 7 or ROCM_VERSION_MINOR < 1 and ROCM_VERSION_MAJOR == 7))
test_edges<__nv_bfloat16>();
#endif
#endif // _LIBCUDACXX_HAS_NVBF16()

  return 0;
}

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

// template<class T>
//   complex<T>
//   polar(const T& rho, const T& theta = T());  // changed from '0' by LWG#2870

#include <cuda/std/cassert>
#include <cuda/std/complex>

#include "../cases.h"
#include "test_macros.h"

template <class T>
__host__ __device__ void test(const T& rho, cuda::std::complex<T> x)
{
  assert(cuda::std::polar(rho) == x);
}

template <class T>
__host__ __device__ void test(const T& rho, const T& theta, cuda::std::complex<T> x)
{
  assert(cuda::std::polar(rho, theta) == x);
}

template <class T>
__host__ __device__ void test()
{
  test(T(0), cuda::std::complex<T>(0, 0));
  test(T(1), cuda::std::complex<T>(1, 0));
  test(T(100), cuda::std::complex<T>(100, 0));
  test(T(0), T(0), cuda::std::complex<T>(0, 0));
  test(T(1), T(0), cuda::std::complex<T>(1, 0));
  test(T(100), T(0), cuda::std::complex<T>(100, 0));
}

template <class T>
__host__ __device__ void test_edges()
{
  auto testcases   = get_testcases<T>();
  const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
  for (unsigned i = 0; i < N; ++i)
  {
    T r                     = real(testcases[i]);
    T theta                 = imag(testcases[i]);
    cuda::std::complex<T> z = cuda::std::polar(r, theta);
    switch (classify(r))
    {
      case zero:
        if (cuda::std::signbit(r) || classify(theta) == inf || classify(theta) == NaN)
        {
          int c = classify(z);
          assert(c == NaN || c == non_zero_nan);
        }
        else
        {
          assert(z == cuda::std::complex<T>());
        }
        break;
      case non_zero:
        if (cuda::std::signbit(r) || classify(theta) == inf || classify(theta) == NaN)
        {
          int c = classify(z);
          assert(c == NaN || c == non_zero_nan);
        }
        else
        {
          printf("in: %f %f\n", float(testcases[i].real()), float(testcases[i].imag()));
          is_about(cuda::std::abs(z), r);
        }
        break;
      case inf:
        if (r < T(0))
        {
          int c = classify(z);
          assert(c == NaN || c == non_zero_nan);
        }
        else
        {
          assert(classify(z) == inf);
          if (classify(theta) != NaN && classify(theta) != inf)
          {
            assert(classify(real(z)) != NaN);
            assert(classify(imag(z)) != NaN);
          }
        }
        break;
      case NaN:
      case non_zero_nan: {
        int c = classify(z);
        assert(c == NaN || c == non_zero_nan);
      }
      break;
    }
  }
}

int main(int, char**)
{
  test<float>();
  test<double>();
// CUDA treats long double as double
//  test<long double>();
#ifdef _LIBCUDACXX_HAS_NVFP16
  test<__half>();
#endif
#ifdef _LIBCUDACXX_HAS_NVBF16
  test<__nv_bfloat16>();
#endif
  test_edges<double>();
#ifdef _LIBCUDACXX_HAS_NVFP16
  test_edges<__half>();
#endif
#ifdef _LIBCUDACXX_HAS_NVBF16
  test_edges<__nv_bfloat16>();
#endif

  return 0;
}

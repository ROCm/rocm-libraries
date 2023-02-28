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
//   T
//   arg(const complex<T>& x);

#include <cuda/std/cassert>
#include <cuda/std/complex>

#include "../cases.h"
#include "test_macros.h"

template <class T>
__host__ __device__ void test()
{
  cuda::std::complex<T> z(1, 0);
  assert(arg(z) == T(0));
}

template <class T>
__host__ __device__ void test_edges()
{
  const T pi       = cuda::std::atan2(+0., -0.);
  auto testcases   = get_testcases<T>();
  const unsigned N = sizeof(testcases) / sizeof(testcases[0]);
  for (unsigned i = 0; i < N; ++i)
  {
    T r = arg(testcases[i]);
    if (cuda::std::isnan(testcases[i].real()) || cuda::std::isnan(testcases[i].imag()))
    {
      assert(cuda::std::isnan(r));
    }
    else
    {
      switch (classify(testcases[i]))
      {
        case zero:
          if (cuda::std::signbit(testcases[i].real()))
          {
            if (cuda::std::signbit(testcases[i].imag()))
            {
              is_about(r, -pi);
            }
            else
            {
              is_about(r, pi);
            }
          }
          else
          {
            assert(cuda::std::signbit(testcases[i].imag()) == cuda::std::signbit(r));
          }
          break;
        case non_zero:
          if (testcases[i].real() == T(0))
          {
            if (testcases[i].imag() < T(0))
            {
              is_about(r, -pi / T(2));
            }
            else
            {
              is_about(r, pi / T(2));
            }
          }
          else if (testcases[i].imag() == T(0))
          {
            if (testcases[i].real() < T(0))
            {
              if (cuda::std::signbit(testcases[i].imag()))
              {
                is_about(r, -pi);
              }
              else
              {
                is_about(r, pi);
              }
            }
            else
            {
              assert(r == T(0));
              assert(cuda::std::signbit(testcases[i].imag()) == cuda::std::signbit(r));
            }
          }
          else if (testcases[i].imag() > T(0))
          {
            assert(r > T(0));
          }
          else
          {
            assert(r < T(0));
          }
          break;
        case inf:
          if (cuda::std::isinf(testcases[i].real()) && cuda::std::isinf(testcases[i].imag()))
          {
            if (testcases[i].real() < T(0))
            {
              if (testcases[i].imag() > T(0))
              {
                is_about(r, T(0.75) * pi);
              }
              else
              {
                is_about(r, T(-0.75) * pi);
              }
            }
            else
            {
              if (testcases[i].imag() > T(0))
              {
                is_about(r, T(0.25) * pi);
              }
              else
              {
                is_about(r, T(-0.25) * pi);
              }
            }
          }
          else if (cuda::std::isinf(testcases[i].real()))
          {
            if (testcases[i].real() < T(0))
            {
              if (cuda::std::signbit(testcases[i].imag()))
              {
                is_about(r, -pi);
              }
              else
              {
                is_about(r, pi);
              }
            }
            else
            {
              assert(r == T(0));
              assert(cuda::std::signbit(r) == cuda::std::signbit(testcases[i].imag()));
            }
          }
          else
          {
            if (testcases[i].imag() < T(0))
            {
              is_about(r, -pi / T(2));
            }
            else
            {
              is_about(r, pi / T(2));
            }
          }
          break;
      }
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

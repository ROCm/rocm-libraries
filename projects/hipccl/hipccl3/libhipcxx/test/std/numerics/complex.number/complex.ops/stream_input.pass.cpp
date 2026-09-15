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

// UNSUPPORTED: no-localization
// UNSUPPORTED: nvrtc, hiprtc

// <complex>

// template<class T, class charT, class traits>
//   basic_istream<charT, traits>&
//   operator>>(basic_istream<charT, traits>& is, complex<T>& x);

#include <cuda/std/cassert>
#include <cuda/std/complex>

#include <sstream>

#include "test_macros.h"

template <class T>
void test()
{
  {
    std::istringstream is("5");
    cuda::std::complex<T> c;
    is >> c;
    assert(c == cuda::std::complex<T>(5, 0));
    assert(is.eof());
  }
  {
    std::istringstream is(" 5 ");
    cuda::std::complex<T> c;
    is >> c;
    assert(c == cuda::std::complex<T>(5, 0));
    assert(is.good());
  }
  {
    std::istringstream is(" 5, ");
    cuda::std::complex<T> c;
    is >> c;
    assert(c == cuda::std::complex<T>(5, 0));
    assert(is.good());
  }
  {
    std::istringstream is(" , 5, ");
    cuda::std::complex<T> c;
    is >> c;
    assert(c == cuda::std::complex<T>(0, 0));
    assert(is.fail());
  }
  {
    std::istringstream is("5.5 ");
    cuda::std::complex<T> c;
    is >> c;
    assert(c == cuda::std::complex<T>(5.5, 0));
    assert(is.good());
  }
  {
    std::istringstream is(" ( 5.5 ) ");
    cuda::std::complex<T> c;
    is >> c;
    assert(c == cuda::std::complex<T>(5.5, 0));
    assert(is.good());
  }
  {
    std::istringstream is("  5.5)");
    cuda::std::complex<T> c;
    is >> c;
    assert(c == cuda::std::complex<T>(5.5, 0));
    assert(is.good());
  }
  {
    std::istringstream is("(5.5 ");
    cuda::std::complex<T> c;
    is >> c;
    assert(c == cuda::std::complex<T>(0, 0));
    assert(is.fail());
  }
  {
    std::istringstream is("(5.5,");
    cuda::std::complex<T> c;
    is >> c;
    assert(c == cuda::std::complex<T>(0, 0));
    assert(is.fail());
  }
  {
    std::istringstream is("( -5.5 , -6.5 )");
    cuda::std::complex<T> c;
    is >> c;
    assert(c == cuda::std::complex<T>(-5.5, -6.5));
    assert(!is.eof());
  }
  {
    std::istringstream is("(-5.5,-6.5)");
    cuda::std::complex<T> c;
    is >> c;
    assert(c == cuda::std::complex<T>(-5.5, -6.5));
    assert(!is.eof());
  }
}

void test()
{
  test<float>();
  test<double>();
#if _LIBCUDACXX_HAS_NVFP16()
  test<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, test();)
  return 0;
}

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

//
// UNSUPPORTED: c++17
// XFAIL: dylib-has-no-filesystem

// Due to C++17 inline variables ASAN flags this test as containing an ODR
// violation because Clock::is_steady is defined in both the dylib and this TU.
// UNSUPPORTED: asan

// File clock is unsupported in NVRTC
// UNSUPPORTED: nvrtc, hiprtc

// <cuda/std/chrono>

// file_clock

// check clock invariants

#include <cuda/std/chrono>

template <class T>
__host__ __device__ void test(const T&)
{}

int main(int, char**)
{
  typedef cuda::std::chrono::file_clock C;
  static_assert((cuda::std::is_same<C::rep, C::duration::rep>::value), "");
  static_assert((cuda::std::is_same<C::period, C::duration::period>::value), "");
  static_assert((cuda::std::is_same<C::duration, C::time_point::duration>::value), "");
  static_assert((cuda::std::is_same<C::time_point::clock, C>::value), "");
  static_assert(!C::is_steady, "");
  test(cuda::std::chrono::file_clock::is_steady);

  return 0;
}

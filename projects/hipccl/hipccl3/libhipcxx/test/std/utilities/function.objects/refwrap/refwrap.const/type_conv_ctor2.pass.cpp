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

// <functional>
//
// reference_wrapper
//
// template <class U>
//   reference_wrapper(U&&);

// #include <cuda/std/functional>
#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "test_macros.h"

struct B
{};

struct A1
{
  mutable B b_;
  __host__ __device__ constexpr operator B&() const
  {
    return b_;
  }
};

struct A2
{
  mutable B b_;
  __host__ __device__ constexpr operator B&() const noexcept
  {
    return b_;
  }
};

__host__ __device__ void implicitly_convert(cuda::std::reference_wrapper<B>) noexcept;

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  {
    A1 a{};
#if !TEST_COMPILER(NVHPC)
    static_assert(!noexcept(implicitly_convert(a)));
#endif // TEST_COMPILER(NVHPC)
    cuda::std::reference_wrapper<B> b1 = a;
    assert(&b1.get() == &a.b_);
#if !TEST_COMPILER(NVHPC)
    static_assert(!noexcept(b1 = a));
#endif // TEST_COMPILER(NVHPC)
    b1 = a;
    assert(&b1.get() == &a.b_);
  }
  {
    A2 a{};
    static_assert(noexcept(implicitly_convert(a)));
    cuda::std::reference_wrapper<B> b2 = a;
    assert(&b2.get() == &a.b_);
    static_assert(noexcept(b2 = a));
    b2 = a;
    assert(&b2.get() == &a.b_);
  }
  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && !TEST_COMPILER(NVRTC) && !defined(TEST_COMPILER_HIPRTC)
  static_assert(test());
#endif // TEST_STD_VER > 2017 && !TEST_COMPILER(NVRTC)

  return 0;
}

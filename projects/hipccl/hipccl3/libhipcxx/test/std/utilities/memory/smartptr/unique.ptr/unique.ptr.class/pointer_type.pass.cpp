//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

// <memory>

// unique_ptr

// Test unique_ptr::pointer type

#include <cuda/std/__memory_>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct Deleter
{
  struct pointer
  {};
};

#if !TEST_COMPILER(GCC) && !TEST_COMPILER(MSVC)
struct D2
{
private:
  typedef void pointer;
};
#endif // !TEST_COMPILER(GCC) && !TEST_COMPILER(MSVC)

#if !TEST_COMPILER(NVRTC) && !defined(TEST_COMPILER_HIPRTC) // A class static data member with non-const type is considered a host variable
struct D3
{
  static long pointer;
};
#endif // !TEST_COMPILER(NVRTC)

template <bool IsArray>
__host__ __device__ TEST_CONSTEXPR_CXX23 void test_basic()
{
  typedef typename cuda::std::conditional<IsArray, int[], int>::type VT;
  {
    typedef cuda::std::unique_ptr<VT> P;
    static_assert((cuda::std::is_same<typename P::pointer, int*>::value), "");
  }
  {
    typedef cuda::std::unique_ptr<VT, Deleter> P;
    static_assert((cuda::std::is_same<typename P::pointer, Deleter::pointer>::value), "");
  }
#if !TEST_COMPILER(GCC) && !TEST_COMPILER(MSVC)
  {
    typedef cuda::std::unique_ptr<VT, D2> P;
    static_assert(cuda::std::is_same<typename P::pointer, int*>::value, "");
  }
#endif // !TEST_COMPILER(GCC) && !TEST_COMPILER(MSVC)
#if !TEST_COMPILER(NVRTC) && !defined(TEST_COMPILER_HIPRTC)
  {
    typedef cuda::std::unique_ptr<VT, D3> P;
    static_assert(cuda::std::is_same<typename P::pointer, int*>::value, "");
  }
#endif // !TEST_COMPILER(NVRTC)
}

__host__ __device__ TEST_CONSTEXPR_CXX23 bool test()
{
  test_basic</*IsArray*/ false>();
  test_basic<true>();

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2023
  static_assert(test());
#endif // TEST_STD_VER >= 2023

  return 0;
}

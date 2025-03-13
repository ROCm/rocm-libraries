//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
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

// <cuda/std/array>

// implicitly generated array constructors / assignment operators

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct NoDefault
{
  __host__ __device__ constexpr NoDefault(int) {}
};

struct NonTrivialCopy
{
  __host__ __device__ constexpr NonTrivialCopy() {}
  __host__ __device__ constexpr NonTrivialCopy(NonTrivialCopy const&) {}
  __host__ __device__ constexpr NonTrivialCopy& operator=(NonTrivialCopy const&)
  {
    return *this;
  }
};

__host__ __device__ constexpr bool tests()
{
  {
    typedef cuda::std::array<double, 3> Array;
    Array array = {1.1, 2.2, 3.3};
    Array copy  = array;
    copy        = array;
    static_assert(cuda::std::is_copy_constructible<Array>::value, "");
    static_assert(cuda::std::is_copy_assignable<Array>::value, "");
    unused(copy);
  }
  {
    typedef cuda::std::array<double const, 3> Array;
    Array array = {1.1, 2.2, 3.3};
    Array copy  = array;
    unused(copy);
    static_assert(cuda::std::is_copy_constructible<Array>::value, "");
    static_assert(!cuda::std::is_copy_assignable<Array>::value, "");
    unused(copy);
  }
  {
    typedef cuda::std::array<double, 0> Array;
    Array array = {};
    Array copy  = array;
    copy        = array;
    static_assert(cuda::std::is_copy_constructible<Array>::value, "");
    static_assert(cuda::std::is_copy_assignable<Array>::value, "");
    unused(copy);
  }
  {
    // const arrays of size 0 should disable the implicit copy assignment operator.
    typedef cuda::std::array<double const, 0> Array;
    Array array = {};
    Array copy  = array;
    static_assert(cuda::std::is_copy_constructible<Array>::value, "");
    static_assert(!cuda::std::is_copy_assignable<Array>::value, "");
    unused(copy);
  }
  {
    typedef cuda::std::array<NoDefault, 0> Array;
    Array array = {};
    Array copy  = array;
    copy        = array;
    static_assert(cuda::std::is_copy_constructible<Array>::value, "");
    static_assert(cuda::std::is_copy_assignable<Array>::value, "");
    unused(copy);
  }
  {
    typedef cuda::std::array<NoDefault const, 0> Array;
    Array array = {};
    Array copy  = array;
    static_assert(cuda::std::is_copy_constructible<Array>::value, "");
    static_assert(!cuda::std::is_copy_assignable<Array>::value, "");
    unused(copy);
  }

  // Make sure we can implicitly copy a cuda::std::array of a non-trivially copyable type
  {
    typedef cuda::std::array<NonTrivialCopy, 0> Array;
    Array array = {};
    Array copy  = array;
    copy        = array;
    static_assert(cuda::std::is_copy_constructible<Array>::value, "");
    unused(copy);
  }

// NVCC believes `copy = array` accesses uninitialized memory
#if TEST_CUDA_COMPILER(NVCC) || TEST_COMPILER(NVRTC) || defined(TEST_COMPILER_HIPCC) || defined(TEST_COMPILER_HIPRTC)
  if (!TEST_IS_CONSTANT_EVALUATED())
#endif // TEST_CUDA_COMPILER(NVCC)
  {
    typedef cuda::std::array<NonTrivialCopy, 1> Array;
    Array array = {};
    Array copy  = array;
    copy        = array;
    static_assert(cuda::std::is_copy_constructible<Array>::value, "");
    unused(copy);
  }
// NVCC believes `copy = array` accesses uninitialized memory
#if TEST_CUDA_COMPILER(NVCC) || TEST_COMPILER(NVRTC) || defined(TEST_COMPILER_HIPCC) || defined(TEST_COMPILER_HIPRTC)
  if (!TEST_IS_CONSTANT_EVALUATED())
#endif // TEST_CUDA_COMPILER(NVCC)
  {
    typedef cuda::std::array<NonTrivialCopy, 2> Array;
    Array array = {};
    Array copy  = array;
    copy        = array;
    static_assert(cuda::std::is_copy_constructible<Array>::value, "");
    unused(copy);
  }

  return true;
}

int main(int, char**)
{
  tests();
#if defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
  static_assert(tests(), "");
#endif
  return 0;
}

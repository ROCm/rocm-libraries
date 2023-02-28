//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
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
// <cuda/std/array>
// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: gcc-6, gcc-7
// UNSUPPORTED: clang-9 && nvcc-11.1

// template <class T, class... U>
//   array(T, U...) -> array<T, 1 + sizeof...(U)>;
//
//  Requires: (is_same_v<T, U> && ...) is true. Otherwise the program is ill-formed.

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>

#include "test_macros.h"

__host__ __device__ constexpr bool tests()
{
  //  Test the explicit deduction guides
  {
    cuda::std::array arr{1, 2, 3}; // array(T, U...)
    static_assert(cuda::std::is_same_v<decltype(arr), cuda::std::array<int, 3>>, "");
    assert(arr[0] == 1);
    assert(arr[1] == 2);
    assert(arr[2] == 3);
  }

  {
    const long l1 = 42;
    cuda::std::array arr{1L, 4L, 9L, l1}; // array(T, U...)
    static_assert(cuda::std::is_same_v<decltype(arr)::value_type, long>, "");
    static_assert(arr.size() == 4, "");
    assert(arr[0] == 1);
    assert(arr[1] == 4);
    assert(arr[2] == 9);
    assert(arr[3] == l1);
  }

  //  Test the implicit deduction guides
  {
    cuda::std::array<double, 2> source = {4.0, 5.0};
    cuda::std::array arr(source); // array(array)
    static_assert(cuda::std::is_same_v<decltype(arr), decltype(source)>, "");
    static_assert(cuda::std::is_same_v<decltype(arr), cuda::std::array<double, 2>>, "");
    assert(arr[0] == 4.0);
    assert(arr[1] == 5.0);
  }

  return true;
}

int main(int, char**)
{
  tests();
  static_assert(tests(), "");
  return 0;
}

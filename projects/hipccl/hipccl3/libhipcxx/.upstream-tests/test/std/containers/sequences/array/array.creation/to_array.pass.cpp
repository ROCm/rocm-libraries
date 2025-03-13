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

// <cuda/std/array>

// UNSUPPORTED: gcc-6, gcc-7, gcc-8

// template <typename T, size_t Size>
// constexpr auto to_array(T (&arr)[Size])
//    -> array<remove_cv_t<T>, Size>;

// template <typename T, size_t Size>
// constexpr auto to_array(T (&&arr)[Size])
//    -> array<remove_cv_t<T>, Size>;

#include <cuda/std/array>
#include <cuda/std/cassert>

#include "MoveOnly.h"
#include "test_macros.h"

__host__ __device__ constexpr bool tests()
{
  //  Test deduced type.
  {
    auto arr = cuda::std::to_array({1, 2, 3});
    static_assert(cuda::std::is_same_v<decltype(arr), cuda::std::array<int, 3>>);
    assert(arr[0] == 1);
    assert(arr[1] == 2);
    assert(arr[2] == 3);
  }

  {
    const long l1 = 42;
    auto arr      = cuda::std::to_array({1L, 4L, 9L, l1});
    static_assert(cuda::std::is_same_v<decltype(arr)::value_type, long>);
    static_assert(arr.size() == 4, "");
    assert(arr[0] == 1);
    assert(arr[1] == 4);
    assert(arr[2] == 9);
    assert(arr[3] == l1);
  }

  {
    auto arr = cuda::std::to_array("meow");
    static_assert(cuda::std::is_same_v<decltype(arr), cuda::std::array<char, 5>>);
    assert(arr[0] == 'm');
    assert(arr[1] == 'e');
    assert(arr[2] == 'o');
    assert(arr[3] == 'w');
    assert(arr[4] == '\0');
  }

  {
    double source[3] = {4.0, 5.0, 6.0};
    auto arr         = cuda::std::to_array(source);
    static_assert(cuda::std::is_same_v<decltype(arr), cuda::std::array<double, 3>>);
    assert(arr[0] == 4.0);
    assert(arr[1] == 5.0);
    assert(arr[2] == 6.0);
  }

  {
    double source[3] = {4.0, 5.0, 6.0};
    auto arr         = cuda::std::to_array(cuda::std::move(source));
    static_assert(cuda::std::is_same_v<decltype(arr), cuda::std::array<double, 3>>);
    assert(arr[0] == 4.0);
    assert(arr[1] == 5.0);
    assert(arr[2] == 6.0);
  }

  {
    MoveOnly source[] = {MoveOnly{0}, MoveOnly{1}, MoveOnly{2}};

    auto arr = cuda::std::to_array(cuda::std::move(source));
    static_assert(cuda::std::is_same_v<decltype(arr), cuda::std::array<MoveOnly, 3>>);
    for (int i = 0; i < 3; ++i)
    {
      assert(arr[i].get() == i && source[i].get() == 0);
    }
  }

#if (TEST_COMPILER(NVRTC) || defined(TEST_COMPILER_HIPRTC)) && TEST_COMPILER(MSVC)
  // Test C99 compound literal.
  {
    auto arr = cuda::std::to_array((int[]) {3, 4});
    static_assert(cuda::std::is_same_v<decltype(arr), cuda::std::array<int, 2>>);
    assert(arr[0] == 3);
    assert(arr[1] == 4);
  }
#endif // !TEST_COMPILER(NVRTC) && !TEST_COMPILER(MSVC)

  //  Test explicit type.
  {
    auto arr = cuda::std::to_array<long>({1, 2, 3});
    static_assert(cuda::std::is_same_v<decltype(arr), cuda::std::array<long, 3>>);
    assert(arr[0] == 1);
    assert(arr[1] == 2);
    assert(arr[2] == 3);
  }

  {
    struct A
    {
      int a;
      double b;
    };

    auto arr = cuda::std::to_array<A>({{3, .1}});
    static_assert(cuda::std::is_same_v<decltype(arr), cuda::std::array<A, 1>>);
    assert(arr[0].a == 3);
    assert(arr[0].b == .1);
  }

  return true;
}

int main(int, char**)
{
  tests();
  static_assert(tests(), "");
  return 0;
}

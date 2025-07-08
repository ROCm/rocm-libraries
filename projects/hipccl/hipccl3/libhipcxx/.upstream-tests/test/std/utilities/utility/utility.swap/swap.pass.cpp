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

// <utility>

// template<class T>
//   requires MoveAssignable<T> && MoveConstructible<T>
//   void
//   swap(T& a, T& b);

#include <cuda/std/__memory_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

#if !defined(TEST_COMPILER_NVRTC) && !defined(TEST_COMPILER_HIPRTC)
#  include <utility>
#endif // !TEST_COMPILER_NVRTC

struct CopyOnly
{
  __host__ __device__ CopyOnly() {}
  __host__ __device__ CopyOnly(CopyOnly const&) noexcept {}
  __host__ __device__ CopyOnly& operator=(CopyOnly const&)
  {
    return *this;
  }
};

struct MoveOnly
{
  __host__ __device__ MoveOnly() {}
  __host__ __device__ MoveOnly(MoveOnly&&) {}
  __host__ __device__ MoveOnly& operator=(MoveOnly&&) noexcept
  {
    return *this;
  }
};

struct NoexceptMoveOnly
{
  __host__ __device__ NoexceptMoveOnly() {}
  __host__ __device__ NoexceptMoveOnly(NoexceptMoveOnly&&) noexcept {}
  __host__ __device__ NoexceptMoveOnly& operator=(NoexceptMoveOnly&&) noexcept
  {
    return *this;
  }
};

struct NotMoveConstructible
{
  __host__ __device__ NotMoveConstructible& operator=(NotMoveConstructible&&)
  {
    return *this;
  }

private:
  __host__ __device__ NotMoveConstructible(NotMoveConstructible&&);
};

struct NotMoveAssignable
{
  __host__ __device__ NotMoveAssignable(NotMoveAssignable&&);

private:
  __host__ __device__ NotMoveAssignable& operator=(NotMoveAssignable&&);
};

template <class Tp>
__host__ __device__ auto
can_swap_test(int) -> decltype(cuda::std::swap(cuda::std::declval<Tp>(), cuda::std::declval<Tp>()));

template <class Tp>
__host__ __device__ auto can_swap_test(...) -> cuda::std::false_type;

template <class Tp>
__host__ __device__ constexpr bool can_swap()
{
  return cuda::std::is_same<decltype(can_swap_test<Tp>(0)), void>::value;
}

#if TEST_STD_VER >= 2014
__host__ __device__ constexpr bool test_swap_constexpr()
{
  int i = 1;
  int j = 2;
  cuda::std::swap(i, j);
  return i == 2 && j == 1;
}
#endif // TEST_STD_VER >= 2014

template <class T>
struct swap_with_friend
{
  __host__ __device__ friend void swap(swap_with_friend&, swap_with_friend&) {}
};

__host__ __device__ void test_ambiguous_std()
{
#if !defined(TEST_COMPILER_NVRTC) && !defined(TEST_COMPILER_HIPRTC)
  // clang-format off
  NV_IF_TARGET(NV_IS_HOST, (
    {
      cuda::std::pair<::std::pair<int, int>, int> i = {};
      cuda::std::pair<::std::pair<int, int>, int> j = {};
      swap(i,j);
    }
    { // Ensure that we do not SFINAE swap out if there is a free function as that will take precedent
      swap_with_friend<::std::pair<int, int>> with_friend;
      cuda::std::swap(with_friend, with_friend);
    }
  ))
  // clang-format on
#  if TEST_STD_VER >= 2014
  static_assert(cuda::std::is_swappable<cuda::std::pair<::std::pair<int, int>, int>>::value, "");
  static_assert(cuda::std::is_swappable<swap_with_friend<::std::pair<int, int>>>::value, "");
#  endif // TEST_STD_VER >= 2014
#endif // !TEST_COMPILER_NVRTC
}

int main(int, char**)
{
  {
    int i = 1;
    int j = 2;
    cuda::std::swap(i, j);
    assert(i == 2);
    assert(j == 1);
  }
  {
    cuda::std::unique_ptr<int> i(new int(1));
    cuda::std::unique_ptr<int> j(new int(2));
    cuda::std::swap(i, j);
    assert(*i == 2);
    assert(*j == 1);
  }
  {
    // test that the swap
    static_assert(can_swap<CopyOnly&>(), "");
    static_assert(can_swap<MoveOnly&>(), "");
    static_assert(can_swap<NoexceptMoveOnly&>(), "");

    static_assert(!can_swap<NotMoveConstructible&>(), "");
    static_assert(!can_swap<NotMoveAssignable&>(), "");

    CopyOnly c;
    MoveOnly m;
    NoexceptMoveOnly nm;
    static_assert(!noexcept(cuda::std::swap(c, c)), "");
    static_assert(!noexcept(cuda::std::swap(m, m)), "");
    static_assert(noexcept(cuda::std::swap(nm, nm)), "");
  }

#if TEST_STD_VER >= 2014
  static_assert(test_swap_constexpr(), "");
#endif // TEST_STD_VER >= 2014

  test_ambiguous_std();

  return 0;
}

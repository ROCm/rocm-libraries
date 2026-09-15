//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// MIT License
//
// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

// <algorithm>

// template<class ForwardIterator>
// constexpr ForwardIterator
//   shift_right(ForwardIterator first, ForwardIterator last,
//               typename iterator_traits<ForwardIterator>::difference_type n);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>

#include "MoveOnly.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class T, class Iter>
__host__ __device__ constexpr void test()
{
  int orig[] = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9};
  T work[]   = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9};

  for (int n = 0; n <= 15; ++n)
  {
    for (int k = 0; k <= n + 2; ++k)
    {
      cuda::std::copy(orig, orig + n, work);
      Iter it = cuda::std::shift_right(Iter(work), Iter(work + n), k);
      if (k < n)
      {
        assert(it == Iter(work + k));
        assert(cuda::std::equal(orig, orig + n - k, work + k, work + n));
      }
      else
      {
        assert(it == Iter(work + n));
        assert(cuda::std::equal(orig, orig + n, work, work + n));
      }
    }
  }

  // n == 0
  {
    T input[]          = {0, 1, 2};
    const T expected[] = {0, 1, 2};
    Iter b             = Iter(cuda::std::begin(input));
    Iter e             = Iter(cuda::std::end(input));
    Iter it            = cuda::std::shift_right(b, e, 0);
    assert(cuda::std::equal(cuda::std::begin(expected), cuda::std::end(expected), it, e));
  }

  // n > 0 && n < len
  {
    T input[]          = {0, 1, 2};
    const T expected[] = {0, 1};
    Iter b             = Iter(cuda::std::begin(input));
    Iter e             = Iter(cuda::std::end(input));
    Iter it            = cuda::std::shift_right(b, e, 1);
    assert(cuda::std::equal(cuda::std::begin(expected), cuda::std::end(expected), it, e));
  }
  {
    T input[]          = {1, 2, 3, 4, 5, 6, 7, 8};
    const T expected[] = {1, 2, 3, 4, 5, 6};
    Iter b             = Iter(cuda::std::begin(input));
    Iter e             = Iter(cuda::std::end(input));
    Iter it            = cuda::std::shift_right(b, e, 2);
    assert(cuda::std::equal(cuda::std::begin(expected), cuda::std::end(expected), it, e));
  }
  {
    T input[]          = {1, 2, 3, 4, 5, 6, 7, 8};
    const T expected[] = {1, 2};
    Iter b             = Iter(cuda::std::begin(input));
    Iter e             = Iter(cuda::std::end(input));
    Iter it            = cuda::std::shift_right(b, e, 6);
    assert(cuda::std::equal(cuda::std::begin(expected), cuda::std::end(expected), it, e));
  }

  // n == len
  {
    constexpr int len     = 3;
    T input[len]          = {0, 1, 2};
    const T expected[len] = {0, 1, 2};
    Iter b                = Iter(cuda::std::begin(input));
    Iter e                = Iter(cuda::std::end(input));
    Iter it               = cuda::std::shift_right(b, e, len);
    assert(cuda::std::equal(cuda::std::begin(expected), cuda::std::end(expected), b, e));
    assert(it == e);
  }

  // n > len
  {
    constexpr int len     = 3;
    T input[len]          = {0, 1, 2};
    const T expected[len] = {0, 1, 2};
    Iter b                = Iter(cuda::std::begin(input));
    Iter e                = Iter(cuda::std::end(input));
    Iter it               = cuda::std::shift_right(b, e, len + 1);
    assert(cuda::std::equal(cuda::std::begin(expected), cuda::std::end(expected), b, e));
    assert(it == e);
  }
}

__host__ __device__ constexpr bool test()
{
  test<int, forward_iterator<int*>>();
  test<int, bidirectional_iterator<int*>>();
  test<int, random_access_iterator<int*>>();
  test<int, int*>();
  test<MoveOnly, forward_iterator<MoveOnly*>>();
  test<MoveOnly, bidirectional_iterator<MoveOnly*>>();
  test<MoveOnly, random_access_iterator<MoveOnly*>>();
  test<MoveOnly, MoveOnly*>();

  return true;
}

int main(int, char**)
{
// NOTE(HIP/AMD): For ROCm 7.2 and HIPRTC we get Illegal instruction detected: Operand has incorrect register class.
// This is fixed in ROCm 7.11+
#if !defined(_CCCL_COMPILER_HIPRTC) and (LIBHIPCXX_ROCM_VERSION_GE(7,11) or LIBHIPCXX_ROCM_VERSION_LE(7,1))
  test();
#if defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
  static_assert(test(), "");
#endif // _CCCL_BUILTIN_IS_CONSTANT_EVALUATED
#endif
  return 0;
}

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

// <cuda/std/functional>

// greater_equal

#define _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

#include <cuda/std/cassert>
#include <cuda/std/functional>
#include <cuda/std/type_traits>

#include "test_macros.h"
#if !defined(TEST_COMPILER_NVRTC) && !defined(TEST_COMPILER_HIPRTC)
#  include "pointer_comparison_test_helper.hpp"
#endif // !TEST_COMPILER(NVRTC)

// ensure that we allow `__device__` functions too
struct with_device_op
{
  __device__ friend constexpr bool operator>=(const with_device_op&, const with_device_op&)
  {
    return true;
  }
};

__global__ void test_global_kernel()
{
  const cuda::std::greater_equal<with_device_op> f;
  assert(f({}, {}));
}

int main(int, char**)
{
  typedef cuda::std::greater_equal<int> F;
  const F f = F();
#if TEST_STD_VER <= 2017
  static_assert((cuda::std::is_same<int, F::first_argument_type>::value), "");
  static_assert((cuda::std::is_same<int, F::second_argument_type>::value), "");
  static_assert((cuda::std::is_same<bool, F::result_type>::value), "");
#endif // TEST_STD_VER <= 2017
  assert(f(36, 36));
  assert(f(36, 6));
  assert(!f(6, 36));
  NV_IF_TARGET(NV_IS_HOST,
               (
                 // test total ordering of int* for greater_equal<int*> and
                 // greater_equal<void>.
                 do_pointer_comparison_test<int, cuda::std::greater_equal>();))

  typedef cuda::std::greater_equal<> F2;
  const F2 f2 = F2();
  assert(f2(36, 36));
  assert(f2(36, 6));
  assert(!f2(6, 36));
  assert(f2(36, 6.0));
  assert(f2(36.0, 6));
  assert(!f2(6, 36.0));
  assert(!f2(6.0, 36));
  constexpr bool foo = cuda::std::greater_equal<int>()(36, 36);
  static_assert(foo, "");

  constexpr bool bar = cuda::std::greater_equal<>()(36.0, 36);
  static_assert(bar, "");

  return 0;
}

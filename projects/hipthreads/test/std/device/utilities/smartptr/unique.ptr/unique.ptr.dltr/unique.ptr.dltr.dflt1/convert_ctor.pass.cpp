//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// default_delete[]

// template <class U>
//   constexpr default_delete(const default_delete<U[]>&); // constexpr since C++23
//
// This constructor shall not participate in overload resolution unless
//   U(*)[] is convertible to T(*)[].

#include "gpu/memory"
#include <cassert>

#include "kernel_launcher.h"
#include "test_macros.h"

__device__ TEST_CONSTEXPR_CXX23 bool test() {
  gpu::default_delete<int[]> d1;
  gpu::default_delete<const int[]> d2 = d1;
  ((void)d2);

  return true;
}

__global__ void gmain() {
  test();
#if TEST_STD_VER >= 23
  static_assert(test());
#endif
}

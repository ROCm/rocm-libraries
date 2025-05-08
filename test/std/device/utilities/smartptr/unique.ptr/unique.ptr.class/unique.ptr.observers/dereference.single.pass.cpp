//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// test op*()

#include "gpu/memory"
#include <cassert>

#include "test_macros.h"
#include "kernel_launcher.h"

__device__ TEST_CONSTEXPR_CXX23 bool test() {
  gpu::unique_ptr<int> p(new int(3));
  assert(*p == 3);

  return true;
}

__global__ void gmain() {
  test();
#if TEST_STD_VER >= 23
  static_assert(test());
#endif

}

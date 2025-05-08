//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// default_delete

#include "gpu/memory"
#include <cassert>

#include "kernel_launcher.h"
#include "test_macros.h"
#include "unique_ptr_test_helper.h"

__device__ TEST_CONSTEXPR_CXX23 bool test() {
  gpu::default_delete<A> d;
  A* p = new A;
  if (!TEST_IS_CONSTANT_EVALUATED)
    assert(A::count == 1);

  d(p);

  if (!TEST_IS_CONSTANT_EVALUATED)
    assert(A::count == 0);

  return true;
}

__global__ void gmain() {
  test();
#if TEST_STD_VER >= 23
  static_assert(test());
#endif
}

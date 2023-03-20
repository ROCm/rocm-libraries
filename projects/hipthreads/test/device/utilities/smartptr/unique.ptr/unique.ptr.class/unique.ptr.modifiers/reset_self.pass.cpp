//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// test reset against resetting self

#include "gpu/memory"

#include "test_macros.h"
#include "kernel_launcher.h"

struct A {
  gpu::unique_ptr<A> ptr_;

  __device__ TEST_CONSTEXPR_CXX23 A() : ptr_(this) {}
  __device__ TEST_CONSTEXPR_CXX23 void reset() { ptr_.reset(); }
};

__device__ TEST_CONSTEXPR_CXX23 bool test() {
  (new A)->reset();

  return true;
}

__global__ void gmain() {
  test();
#if TEST_STD_VER >= 23
  static_assert(test());
#endif

}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// test op[](size_t)

#include "gpu/memory"
#include <cassert>

// TODO: Move TEST_IS_CONSTANT_EVALUATED into it's own header
#include <type_traits>

#include "test_macros.h"
#include "kernel_launcher.h"

class A {
  int state_;
  static __device__ int next_;

public:
  __device__ TEST_CONSTEXPR_CXX23 A() : state_(0) {
    if (!TEST_IS_CONSTANT_EVALUATED)
      state_ = ++next_;
  }

  __device__ TEST_CONSTEXPR_CXX23 int get() const { return state_; }

  __device__ friend TEST_CONSTEXPR_CXX23 bool operator==(const A& x, int y) { return x.state_ == y; }

  __device__ TEST_CONSTEXPR_CXX23 A& operator=(int i) {
    state_ = i;
    return *this;
  }
};

int A::next_ = 0;

__device__ TEST_CONSTEXPR_CXX23 bool test() {
  gpu::unique_ptr<A[]> p(new A[3]);
  if (!TEST_IS_CONSTANT_EVALUATED) {
    assert(p[0] == 1);
    assert(p[1] == 2);
    assert(p[2] == 3);
  }
  p[0] = 3;
  p[1] = 2;
  p[2] = 1;
  assert(p[0] == 3);
  assert(p[1] == 2);
  assert(p[2] == 1);

  return true;
}

__global__ void gmain() {
  test();
#if TEST_STD_VER >= 23
  static_assert(test());
#endif

}

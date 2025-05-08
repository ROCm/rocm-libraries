//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
#include "gpu/memory"
#include <string>
#include <cassert>

#include "kernel_launcher.h"
#include "test_macros.h"

//    The only way to create an unique_ptr<T[]> is to default construct them.

class foo {
public:
  __device__ TEST_CONSTEXPR_CXX23 foo() : val_(3) {}
  __device__ TEST_CONSTEXPR_CXX23 int get() const { return val_; }

private:
  int val_;
};

__device__ TEST_CONSTEXPR_CXX23 bool test() {
  {
    auto p1 = gpu::make_unique<int[]>(5);
    for (int i = 0; i < 5; ++i)
      assert(p1[i] == 0);
  }

  // TODO: uncomment once we implement gpu::string
  // {
  //   auto p2 = gpu::make_unique<gpu::string[]>(5);
  //   for (int i = 0; i < 5; ++i)
  //     assert(p2[i].size() == 0);
  // }

  {
    auto p3 = gpu::make_unique<foo[]>(7);
    for (int i = 0; i < 7; ++i)
      assert(p3[i].get() == 3);
  }

  return true;
}

__global__ void gmain() {
  test();
#if TEST_STD_VER >= 23
  static_assert(test());
#endif
}

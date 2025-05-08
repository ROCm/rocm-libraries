//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
#include "gpu/memory"
//#include <string>
#include <cassert>

#include "kernel_launcher.h"
#include "test_macros.h"

__device__ TEST_CONSTEXPR_CXX23 bool test() {
  {
    gpu::unique_ptr<int> p1 = gpu::make_unique<int>(1);
    assert(*p1 == 1);
    p1 = gpu::make_unique<int>();
    assert(*p1 == 0);
  }

  // {
  //   gpu::unique_ptr<gpu::string> p2 = gpu::make_unique<gpu::string>("Meow!");
  //   assert(*p2 == "Meow!");
  //   p2 = gpu::make_unique<gpu::string>();
  //   assert(*p2 == "");
  //   p2 = gpu::make_unique<gpu::string>(6, 'z');
  //   assert(*p2 == "zzzzzz");
  // }

  return true;
}

__global__ void gmain() {
  test();
#if TEST_STD_VER >= 23
  static_assert(test());
#endif
}

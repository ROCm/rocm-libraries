//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// test op->()

#include "gpu/memory"
#include <cassert>

#include "test_macros.h"

struct A {
  int i_;

  TEST_CONSTEXPR_CXX23 A() : i_(7) {}
};

TEST_CONSTEXPR_CXX23 bool test() {
  gpu::unique_ptr_h<A> p = gpu::make_unique<A>(A());
  assert(p->i_ == 7);

  return true;
}

int main(int, char **) {
  test();
#if TEST_STD_VER >= 23
  static_assert(test());
#endif

}

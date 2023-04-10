//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// test reset

#include "gpu/memory"
#include <cassert>

#include "test_macros.h"
#include "unique_ptr_test_helper.h"

TEST_CONSTEXPR_CXX23 bool test() {
  {
    gpu::unique_ptr_h<A_h> p(newValue<A_h>(1));
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(A_h::count == 1);
      assert(B_h::count == 0);
    }
    A_h* i = p.get();
    assert(i != nullptr);
    p.reset(newValue<B_h>(1));
    // In order to get A_h and B_h to be trivially destrictible, we have to use a hack where Deleter is the one
    // decrementing count, but that doesn't work for unique_ptr_h
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(--A_h::count == 1);
      assert(B_h::count == 1);
    }
  }
  // In order to get A_h and B_h to be trivially destrictible, we have to use a hack where Deleter is the one
  // decrementing count, but that doesn't work for unique_ptr_h
  if (!TEST_IS_CONSTANT_EVALUATED) {
    assert(--A_h::count == 0);
    assert(--B_h::count == 0);
  }
  {
    gpu::unique_ptr_h<A_h> p(newValue<B_h>(1));
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(A_h::count == 1);
      assert(B_h::count == 1);
    }
    A_h* i = p.get();
    assert(i != nullptr);
    p.reset(newValue<B_h>(1));
    // In order to get A_h and B_h to be trivially destrictible, we have to use a hack where Deleter is the one
    // decrementing count, but that doesn't work for unique_ptr_h
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(--A_h::count == 1);
      assert(--B_h::count == 1);
    }
  }
  if (!TEST_IS_CONSTANT_EVALUATED) {
    assert(--A_h::count == 0);
    assert(--B_h::count == 0);
  }

  return true;
}

int main(int, char **) {
  test();
#if TEST_STD_VER >= 23
  static_assert(test());
#endif

  return 0;
}

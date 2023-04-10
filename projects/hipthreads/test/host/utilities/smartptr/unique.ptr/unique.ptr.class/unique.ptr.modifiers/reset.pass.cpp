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

template <bool IsArray>
TEST_CONSTEXPR_CXX23 void test_reset_pointer() {
  typedef typename std::conditional<IsArray, A_h[], A_h>::type VT;
  const int expect_alive = IsArray ? 3 : 1;
#if TEST_STD_VER >= 11
  {
    using U = gpu::unique_ptr_h<VT>;
    U u;
    ((void)u);
    ASSERT_NOEXCEPT(u.reset((A_h*)nullptr));
  }
#endif
  {
    gpu::unique_ptr_h<VT> p(newValue<VT>(expect_alive));
    if (!TEST_IS_CONSTANT_EVALUATED)
      assert(A_h::count == expect_alive);
    A_h* i = p.get();
    assert(i != nullptr);
    A_h* new_value = newValue<VT>(expect_alive);
    if (!TEST_IS_CONSTANT_EVALUATED)
      assert(A_h::count == (expect_alive * 2));
    p.reset(new_value);
    // In order to get A_h and B_h to be trivially destrictible, we have to use a hack where Deleter is the one
    // decrementing count, but that doesn't work for unique_ptr_h
    if (!TEST_IS_CONSTANT_EVALUATED)
      assert((A_h::count -= expect_alive) == expect_alive);
    assert(p.get() == new_value);
  }
  // In order to get A_h and B_h to be trivially destrictible, we have to use a hack where Deleter is the one
  // decrementing count, but that doesn't work for unique_ptr_h
  if (!TEST_IS_CONSTANT_EVALUATED)
    assert((A_h::count -= expect_alive) == 0);
  {
    gpu::unique_ptr_h<const VT> p(newValue<const VT>(expect_alive));
    if (!TEST_IS_CONSTANT_EVALUATED)
      assert(A_h::count == expect_alive);
    const A_h* i = p.get();
    assert(i != nullptr);
    A_h* new_value = newValue<VT>(expect_alive);
    if (!TEST_IS_CONSTANT_EVALUATED)
      assert(A_h::count == (expect_alive * 2));
    p.reset(new_value);
    // In order to get A_h and B_h to be trivially destrictible, we have to use a hack where Deleter is the one
    // decrementing count, but that doesn't work for unique_ptr_h
    if (!TEST_IS_CONSTANT_EVALUATED)
      assert((A_h::count -= expect_alive) == expect_alive);
    assert(p.get() == new_value);
  }
  // In order to get A_h and B_h to be trivially destrictible, we have to use a hack where Deleter is the one
  // decrementing count, but that doesn't work for unique_ptr_h
  if (!TEST_IS_CONSTANT_EVALUATED)
    assert((A_h::count -= expect_alive) == 0);
}

template <bool IsArray>
TEST_CONSTEXPR_CXX23 void test_reset_nullptr() {
  typedef typename std::conditional<IsArray, A_h[], A_h>::type VT;
  const int expect_alive = IsArray ? 3 : 1;
#if TEST_STD_VER >= 11
  {
    using U = gpu::unique_ptr_h<VT>;
    U u;
    ((void)u);
    ASSERT_NOEXCEPT(u.reset(nullptr));
  }
#endif
  {
    gpu::unique_ptr_h<VT> p(newValue<VT>(expect_alive));
    if (!TEST_IS_CONSTANT_EVALUATED)
      assert(A_h::count == expect_alive);
    A_h* i = p.get();
    assert(i != nullptr);
    p.reset(nullptr);
    // In order to get A_h and B_h to be trivially destrictible, we have to use a hack where Deleter is the one
    // decrementing count, but that doesn't work for unique_ptr_h
    if (!TEST_IS_CONSTANT_EVALUATED)
      assert((A_h::count -= expect_alive) == 0);
    assert(p.get() == nullptr);
  }
  if (!TEST_IS_CONSTANT_EVALUATED)
    assert(A_h::count == 0);
}

template <bool IsArray>
TEST_CONSTEXPR_CXX23 void test_reset_no_arg() {
  typedef typename std::conditional<IsArray, A_h[], A_h>::type VT;
  const int expect_alive = IsArray ? 3 : 1;
#if TEST_STD_VER >= 11
  {
    using U = gpu::unique_ptr_h<VT>;
    U u;
    ((void)u);
    ASSERT_NOEXCEPT(u.reset());
  }
#endif
  {
    gpu::unique_ptr_h<VT> p(newValue<VT>(expect_alive));
    if (!TEST_IS_CONSTANT_EVALUATED)
      assert(A_h::count == expect_alive);
    A_h* i = p.get();
    assert(i != nullptr);
    p.reset();
    // In order to get A_h and B_h to be trivially destrictible, we have to use a hack where Deleter is the one
    // decrementing count, but that doesn't work for unique_ptr_h
    if (!TEST_IS_CONSTANT_EVALUATED)
      assert((A_h::count -= expect_alive) == 0);
    assert(p.get() == nullptr);
  }
  if (!TEST_IS_CONSTANT_EVALUATED)
    assert(A_h::count == 0);
}

TEST_CONSTEXPR_CXX23 bool test() {
  {
    test_reset_pointer</*IsArray*/ false>();
    test_reset_nullptr<false>();
    test_reset_no_arg<false>();
  }
  {
    test_reset_pointer</*IsArray*/true>();
    test_reset_nullptr<true>();
    test_reset_no_arg<true>();
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

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <memory>

// unique_ptr

// Test unique_ptr converting move assignment

#include "gpu/memory"
#include <utility>
#include <cassert>

#include "test_macros.h"
#include "deleter_types.h"
#include "unique_ptr_test_helper.h"

template <class APtr, class BPtr>
TEST_CONSTEXPR_CXX23 void testAssign(APtr& aptr, BPtr& bptr) {
  A_h* p = bptr.get();
  if (!TEST_IS_CONSTANT_EVALUATED)
    assert(A_h::count == 2);
  aptr = std::move(bptr);
  assert(aptr.get() == p);
  assert(bptr.get() == 0);
  // In order for A_h to be trivially copyable, we had to get rid of the destructor
  if (!TEST_IS_CONSTANT_EVALUATED) {
    assert(A_h::count == 1);
    assert(B_h::count == 1);
  }
}

template <class LHS, class RHS>
TEST_CONSTEXPR_CXX23 void checkDeleter(LHS& lhs, RHS& rhs, int LHSState, int RHSState) {
  assert(lhs.get_deleter().state() == LHSState);
  assert(rhs.get_deleter().state() == RHSState);
}

template <class T>
struct NCConvertingDeleter {
  TEST_CONSTEXPR_CXX23 NCConvertingDeleter()                      = default;
  NCConvertingDeleter(NCConvertingDeleter const&) = delete;
  TEST_CONSTEXPR_CXX23 NCConvertingDeleter(NCConvertingDeleter&&) = default;

  template <class U>
  TEST_CONSTEXPR_CXX23 NCConvertingDeleter(NCConvertingDeleter<U>&&) {}

  TEST_CONSTEXPR_CXX23 void operator()(T*) const {}
};

template <class T>
struct NCConvertingDeleter<T[]> {
  TEST_CONSTEXPR_CXX23 NCConvertingDeleter()                      = default;
  NCConvertingDeleter(NCConvertingDeleter const&) = delete;
  TEST_CONSTEXPR_CXX23 NCConvertingDeleter(NCConvertingDeleter&&) = default;

  template <class U>
  TEST_CONSTEXPR_CXX23 NCConvertingDeleter(NCConvertingDeleter<U>&&) {}

  TEST_CONSTEXPR_CXX23 void operator()(T*) const {}
};

struct NCGenericDeleter {
  TEST_CONSTEXPR_CXX23 NCGenericDeleter()                   = default;
  NCGenericDeleter(NCGenericDeleter const&) = delete;
  TEST_CONSTEXPR_CXX23 NCGenericDeleter(NCGenericDeleter&&) = default;

  TEST_CONSTEXPR_CXX23 void operator()(void*) const {}
};

TEST_CONSTEXPR_CXX23 void test_sfinae() {
  using DA = NCConvertingDeleter<A>; // non-copyable deleters
  using DB = NCConvertingDeleter<B>;
  using UA = gpu::unique_ptr<A>;
  using UB = gpu::unique_ptr<B>;
  using UAD = gpu::unique_ptr<A, DA>;
  using UBD = gpu::unique_ptr<B, DB>;
  { // cannot move from an lvalue
    static_assert(std::is_assignable<UA, UB&&>::value, "");
    static_assert(!std::is_assignable<UA, UB&>::value, "");
    static_assert(!std::is_assignable<UA, const UB&>::value, "");
  }
  { // cannot move if the deleter-types cannot convert
    static_assert(std::is_assignable<UAD, UBD&&>::value, "");
    static_assert(!std::is_assignable<UAD, UB&&>::value, "");
    static_assert(!std::is_assignable<UA, UBD&&>::value, "");
  }
  { // cannot move-convert with reference deleters of different types
    using UA1 = gpu::unique_ptr<A, DA&>;
    using UB1 = gpu::unique_ptr<B, DB&>;
    static_assert(!std::is_assignable<UA1, UB1&&>::value, "");
  }
  { // cannot move-convert with reference deleters of different types
    using UA1 = gpu::unique_ptr<A, const DA&>;
    using UB1 = gpu::unique_ptr<B, const DB&>;
    static_assert(!std::is_assignable<UA1, UB1&&>::value, "");
  }
  { // cannot move-convert from unique_ptr<Array[]>
    using UA1 = gpu::unique_ptr<A>;
    using UA2 = gpu::unique_ptr<A[]>;
    using UB1 = gpu::unique_ptr<B[]>;
    static_assert(!std::is_assignable<UA1, UA2&&>::value, "");
    static_assert(!std::is_assignable<UA1, UB1&&>::value, "");
  }
  { // cannot move-convert from unique_ptr<Array[]>
    using UA1 = gpu::unique_ptr<A, NCGenericDeleter>;
    using UA2 = gpu::unique_ptr<A[], NCGenericDeleter>;
    using UB1 = gpu::unique_ptr<B[], NCGenericDeleter>;
    static_assert(!std::is_assignable<UA1, UA2&&>::value, "");
    static_assert(!std::is_assignable<UA1, UB1&&>::value, "");
  }
}

TEST_CONSTEXPR_CXX23 bool test() {
  test_sfinae();
  // {
  //   gpu::unique_ptr<B> bptr(new B);
  //   gpu::unique_ptr<A> aptr(new A);
  //   testAssign(aptr, bptr);
  // }
  if (!TEST_IS_CONSTANT_EVALUATED) {
    assert(A_h::count == 0);
    assert(B_h::count == 0);
  }
  {
    Deleter<B_h> del(42);
    gpu::unique_ptr<B_h, Deleter<B_h> > bptr(newValue<B_h>(1), std::move(del));
    gpu::unique_ptr<A_h, Deleter<A_h> > aptr(newValue<A_h>(1));
    testAssign(aptr, bptr);
    checkDeleter(aptr, bptr, 42, 0);
  }
  if (!TEST_IS_CONSTANT_EVALUATED) {
    assert(A_h::count == 0);
    // In order to get A_h and B_h to be trivially destrictible, we have to use a hack where Deleter is the one decrementing count
    // But it doesn't work quite right for polymorphic types, so we need to manually decrement B_h::count here 
    assert(--B_h::count == 0);
  }
  {
    CDeleter<A_h> adel(6);
    CDeleter<B_h> bdel(42);
    gpu::unique_ptr<B_h, CDeleter<B_h>&> bptr(newValue<B_h>(1), bdel);
    gpu::unique_ptr<A_h, CDeleter<A_h>&> aptr(newValue<A_h>(1), adel);
    testAssign(aptr, bptr);
    checkDeleter(aptr, bptr, 42, 42);
  }
  if (!TEST_IS_CONSTANT_EVALUATED) {
    assert(A_h::count == 0);
    // In order to get A_h and B_h to be trivially destrictible, we have to use a hack where Deleter is the one decrementing count
    // But it doesn't work quite right for polymorphic types, so we need to manually decrement B_h::count here 
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

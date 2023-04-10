//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// Test unique_ptr converting move ctor

// NOTE: unique_ptr does not provide converting constructors in C++03
// UNSUPPORTED: c++03

#include "gpu/memory"
#include <type_traits>
#include <utility>
#include <cassert>

#include "test_macros.h"
#include "unique_ptr_test_helper.h"

// test converting move ctor.  Should only require a MoveConstructible deleter, or if
//    deleter is a reference, not even that.
// Explicit version

template <class LHS, class RHS>
TEST_CONSTEXPR_CXX23 void checkReferenceDeleter(LHS& lhs, RHS& rhs) {
  typedef typename LHS::deleter_type NewDel;
  static_assert(std::is_reference<NewDel>::value, "");
  rhs.get_deleter().set_state(42);
  assert(rhs.get_deleter().state() == 42);
  assert(lhs.get_deleter().state() == 42);
  lhs.get_deleter().set_state(99);
  assert(lhs.get_deleter().state() == 99);
  assert(rhs.get_deleter().state() == 99);
}

template <class LHS, class RHS>
TEST_CONSTEXPR_CXX23 void checkDeleter(LHS& lhs, RHS& rhs, int LHSVal, int RHSVal) {
  assert(lhs.get_deleter().state() == LHSVal);
  assert(rhs.get_deleter().state() == RHSVal);
}

template <class LHS, class RHS>
TEST_CONSTEXPR_CXX23 void checkCtor(LHS& lhs, RHS& rhs, A_h* RHSVal) {
  assert(lhs.get() == RHSVal);
  assert(rhs.get() == nullptr);
  // TODO: 
  // if (!TEST_IS_CONSTANT_EVALUATED) {
  //   assert(A_h::count == 1);
  //   assert(B_h::count == 1);
  // }
}

TEST_CONSTEXPR_CXX23 void checkNoneAlive() {
  // TODO:
  // if (!TEST_IS_CONSTANT_EVALUATED) {
  //   assert(A_h::count == 0);
  //   assert(B_h::count == 0);
  // }
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
  using DA = NCConvertingDeleter<A_h>; // non-copyable deleters
  using DB = NCConvertingDeleter<B_h>;
  using UA = gpu::unique_ptr_h<A_h>;
  using UB = gpu::unique_ptr_h<B_h>;
  using UAD = gpu::unique_ptr<A_h, DA>;
  using UBD = gpu::unique_ptr<B_h, DB>;
  { // cannot move from an lvalue
    static_assert(std::is_constructible<UA, UB&&>::value, "");
    static_assert(!std::is_constructible<UA, UB&>::value, "");
    static_assert(!std::is_constructible<UA, const UB&>::value, "");
  }
  { // cannot move if the deleter-types cannot convert
    static_assert(std::is_constructible<UAD, UBD&&>::value, "");
    static_assert(!std::is_constructible<UAD, UB&&>::value, "");
    static_assert(!std::is_constructible<UA, UBD&&>::value, "");
  }
  { // cannot move-convert with reference deleters of different types
    using UA1 = gpu::unique_ptr<A_h, DA&>;
    using UB1 = gpu::unique_ptr<B_h, DB&>;
    static_assert(!std::is_constructible<UA1, UB1&&>::value, "");
  }
  { // cannot move-convert with reference deleters of different types
    using UA1 = gpu::unique_ptr<A_h, const DA&>;
    using UB1 = gpu::unique_ptr<B_h, const DB&>;
    static_assert(!std::is_constructible<UA1, UB1&&>::value, "");
  }
  { // cannot move-convert from unique_ptr<Array[]>
    using UA1 = gpu::unique_ptr_h<A_h>;
    using UA2 = gpu::unique_ptr_h<A_h[]>;
    using UB1 = gpu::unique_ptr_h<B_h[]>;
    static_assert(!std::is_constructible<UA1, UA2&&>::value, "");
    static_assert(!std::is_constructible<UA1, UB1&&>::value, "");
  }
  { // cannot move-convert from unique_ptr<Array[]>
    using UA1 = gpu::unique_ptr<A_h, NCGenericDeleter>;
    using UA2 = gpu::unique_ptr<A_h[], NCGenericDeleter>;
    using UB1 = gpu::unique_ptr<B_h[], NCGenericDeleter>;
    static_assert(!std::is_constructible<UA1, UA2&&>::value, "");
    static_assert(!std::is_constructible<UA1, UB1&&>::value, "");
  }
}

TEST_CONSTEXPR_CXX23 void test_noexcept() {
  {
    typedef gpu::unique_ptr_h<A_h> APtr;
    typedef gpu::unique_ptr_h<B_h> BPtr;
    static_assert(std::is_nothrow_constructible<APtr, BPtr>::value, "");
  }
  {
    typedef gpu::unique_ptr<A_h, Deleter<A_h> > APtr;
    typedef gpu::unique_ptr<B_h, Deleter<B_h> > BPtr;
    static_assert(std::is_nothrow_constructible<APtr, BPtr>::value, "");
  }
  {
    typedef gpu::unique_ptr<A_h, NCDeleter<A_h>&> APtr;
    typedef gpu::unique_ptr<B_h, NCDeleter<A_h>&> BPtr;
    static_assert(std::is_nothrow_constructible<APtr, BPtr>::value, "");
  }
  {
    typedef gpu::unique_ptr<A_h, const NCConstDeleter<A_h>&> APtr;
    typedef gpu::unique_ptr<B_h, const NCConstDeleter<A_h>&> BPtr;
    static_assert(std::is_nothrow_constructible<APtr, BPtr>::value, "");
  }
}

TEST_CONSTEXPR_CXX23 bool test() {
  {
    test_sfinae();
    test_noexcept();
  }
  {
    typedef gpu::unique_ptr_h<A_h> APtr;
    typedef gpu::unique_ptr_h<B_h> BPtr;
    { // explicit
      BPtr b(newValue<B_h>(1));
      A_h* p = b.get();
      APtr a(std::move(b));
      checkCtor(a, b, p);
    }
    checkNoneAlive();
    { // implicit
      BPtr b(newValue<B_h>(1));
      A_h* p = b.get();
      APtr a = std::move(b);
      checkCtor(a, b, p);
    }
    checkNoneAlive();
  }
  { // test with moveable deleters
    typedef gpu::unique_ptr<A_h, Deleter<A_h> > APtr;
    typedef gpu::unique_ptr<B_h, Deleter<B_h> > BPtr;
    {
      Deleter<B_h> del(5);
      BPtr b(newValue<B_h>(1), std::move(del));
      A_h* p = b.get();
      APtr a(std::move(b));
      checkCtor(a, b, p);
      checkDeleter(a, b, 5, 0);
    }
    checkNoneAlive();
    {
      Deleter<B_h> del(5);
      BPtr b(newValue<B_h>(1), std::move(del));
      A_h* p = b.get();
      APtr a = std::move(b);
      checkCtor(a, b, p);
      checkDeleter(a, b, 5, 0);
    }
    checkNoneAlive();
  }
  { // test with reference deleters
    typedef gpu::unique_ptr<A_h, NCDeleter<A_h>&> APtr;
    typedef gpu::unique_ptr<B_h, NCDeleter<A_h>&> BPtr;
    NCDeleter<A_h> del(5);
    {
      BPtr b(newValue<B_h>(1), del);
      A_h* p = b.get();
      APtr a(std::move(b));
      checkCtor(a, b, p);
      checkReferenceDeleter(a, b);
    }
    checkNoneAlive();
    {
      BPtr b(newValue<B_h>(1), del);
      A_h* p = b.get();
      APtr a = std::move(b);
      checkCtor(a, b, p);
      checkReferenceDeleter(a, b);
    }
    checkNoneAlive();
  }
  {
    typedef gpu::unique_ptr<A_h, CDeleter<A_h> > APtr;
    typedef gpu::unique_ptr<B_h, CDeleter<B_h>&> BPtr;
    CDeleter<B_h> del(5);
    {
      BPtr b(newValue<B_h>(1), del);
      A_h* p = b.get();
      APtr a(std::move(b));
      checkCtor(a, b, p);
      checkDeleter(a, b, 5, 5);
    }
    checkNoneAlive();
    {
      BPtr b(newValue<B_h>(1), del);
      A_h* p = b.get();
      APtr a = std::move(b);
      checkCtor(a, b, p);
      checkDeleter(a, b, 5, 5);
    }
    checkNoneAlive();
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

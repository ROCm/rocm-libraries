//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

//=============================================================================
// TESTING unique_ptr(pointer, deleter)
//
// Concerns:
//   1 unique_ptr(pointer, deleter&&) only requires a MoveConstructible deleter.
//   2 unique_ptr(pointer, deleter&) requires a CopyConstructible deleter.
//   3 unique_ptr<T, D&>(pointer, deleter) does not require a CopyConstructible deleter.
//   4 unique_ptr<T, D const&>(pointer, deleter) does not require a CopyConstructible deleter.
//   5 unique_ptr(pointer, deleter) should work for derived pointers.
//   6 unique_ptr(pointer, deleter) should work with function pointers.
//   7 unique_ptr<void> should work.

#include "gpu/memory"
#include <cassert>

#include "test_macros.h"
#include "unique_ptr_test_helper.h"

bool my_free_called = false;

void my_free(void*) { my_free_called = true; }

#if TEST_STD_VER >= 11
struct DeleterBase {
  TEST_CONSTEXPR_CXX23 void operator()(void*) const {}
};
struct CopyOnlyDeleter : DeleterBase {
  TEST_CONSTEXPR_CXX23 CopyOnlyDeleter()                       = default;
  TEST_CONSTEXPR_CXX23 CopyOnlyDeleter(CopyOnlyDeleter const&) = default;
  CopyOnlyDeleter(CopyOnlyDeleter&&) = delete;
};
struct MoveOnlyDeleter : DeleterBase {
  TEST_CONSTEXPR_CXX23 MoveOnlyDeleter()                  = default;
  TEST_CONSTEXPR_CXX23 MoveOnlyDeleter(MoveOnlyDeleter&&) = default;
};
struct NoCopyMoveDeleter : DeleterBase {
  TEST_CONSTEXPR_CXX23 NoCopyMoveDeleter()    = default;
  NoCopyMoveDeleter(NoCopyMoveDeleter const&) = delete;
};
#endif

template <bool IsArray>
TEST_CONSTEXPR_CXX23 void test_sfinae() {
#if TEST_STD_VER >= 11
  typedef typename std::conditional<!IsArray, int, int[]>::type VT;
  {
    using D = CopyOnlyDeleter;
    using U = gpu::unique_ptr<VT, D>;
    static_assert(std::is_constructible<U, int*, D const&>::value, "");
    static_assert(std::is_constructible<U, int*, D&>::value, "");
    static_assert(std::is_constructible<U, int*, D&&>::value, "");
    // FIXME: __libcpp_compressed_pair attempts to perform a move even though
    // it should only copy.
    //D d;
    //U u(nullptr, std::move(d));
  }
  {
    using D = MoveOnlyDeleter;
    using U = gpu::unique_ptr<VT, D>;
    static_assert(!std::is_constructible<U, int*, D const&>::value, "");
    static_assert(!std::is_constructible<U, int*, D&>::value, "");
    static_assert(std::is_constructible<U, int*, D&&>::value, "");
    D d;
    U u(nullptr, std::move(d));
  }
  {
    using D = NoCopyMoveDeleter;
    using U = gpu::unique_ptr<VT, D>;
    static_assert(!std::is_constructible<U, int*, D const&>::value, "");
    static_assert(!std::is_constructible<U, int*, D&>::value, "");
    static_assert(!std::is_constructible<U, int*, D&&>::value, "");
  }
  {
    using D = NoCopyMoveDeleter;
    using U = gpu::unique_ptr<VT, D&>;
    static_assert(!std::is_constructible<U, int*, D const&>::value, "");
    static_assert(std::is_constructible<U, int*, D&>::value, "");
    static_assert(!std::is_constructible<U, int*, D&&>::value, "");
    static_assert(!std::is_constructible<U, int*, const D&&>::value, "");
  }
  {
    using D = NoCopyMoveDeleter;
    using U = gpu::unique_ptr<VT, const D&>;
    static_assert(std::is_constructible<U, int*, D const&>::value, "");
    static_assert(std::is_constructible<U, int*, D&>::value, "");
    static_assert(!std::is_constructible<U, int*, D&&>::value, "");
    static_assert(!std::is_constructible<U, int*, const D&&>::value, "");
  }
#endif
}

template <bool IsArray>
TEST_CONSTEXPR_CXX23 void test_noexcept() {
#if TEST_STD_VER >= 11
  typedef typename std::conditional<!IsArray, int, int[]>::type VT;
  {
    using D = CopyOnlyDeleter;
    using U = gpu::unique_ptr<VT, D>;
    static_assert(std::is_nothrow_constructible<U, int*, D const&>::value, "");
    static_assert(std::is_nothrow_constructible<U, int*, D&>::value, "");
    static_assert(std::is_nothrow_constructible<U, int*, D&&>::value, "");
  }
  {
    using D = MoveOnlyDeleter;
    using U = gpu::unique_ptr<VT, D>;
    static_assert(std::is_nothrow_constructible<U, int*, D&&>::value, "");
    D d;
    U u(nullptr, std::move(d));
  }
  {
    using D = NoCopyMoveDeleter;
    using U = gpu::unique_ptr<VT, D&>;
    static_assert(std::is_nothrow_constructible<U, int*, D&>::value, "");
  }
  {
    using D = NoCopyMoveDeleter;
    using U = gpu::unique_ptr<VT, const D&>;
    static_assert(std::is_nothrow_constructible<U, int*, D const&>::value, "");
    static_assert(std::is_nothrow_constructible<U, int*, D&>::value, "");
  }
#endif
}

TEST_CONSTEXPR_CXX23 void test_sfinae_runtime() {
#if TEST_STD_VER >= 11
  {
    using D = CopyOnlyDeleter;
    using U = gpu::unique_ptr<A_h[], D>;
    static_assert(std::is_nothrow_constructible<U, A_h*, D const&>::value, "");
    static_assert(std::is_nothrow_constructible<U, A_h*, D&>::value, "");
    static_assert(std::is_nothrow_constructible<U, A_h*, D&&>::value, "");

    static_assert(!std::is_constructible<U, B_h*, D const&>::value, "");
    static_assert(!std::is_constructible<U, B_h*, D&>::value, "");
    static_assert(!std::is_constructible<U, B_h*, D&&>::value, "");
    // FIXME: __libcpp_compressed_pair attempts to perform a move even though
    // it should only copy.
    //D d;
    //U u(nullptr, std::move(d));
  }
  {
    using D = MoveOnlyDeleter;
    using U = gpu::unique_ptr<A_h[], D>;
    static_assert(!std::is_constructible<U, A_h*, D const&>::value, "");
    static_assert(!std::is_constructible<U, A_h*, D&>::value, "");
    static_assert(std::is_nothrow_constructible<U, A_h*, D&&>::value, "");

    static_assert(!std::is_constructible<U, B_h*, D const&>::value, "");
    static_assert(!std::is_constructible<U, B_h*, D&>::value, "");
    static_assert(!std::is_constructible<U, B_h*, D&&>::value, "");
    D d;
    U u(nullptr, std::move(d));
  }
  {
    using D = NoCopyMoveDeleter;
    using U = gpu::unique_ptr<A_h[], D>;
    static_assert(!std::is_constructible<U, A_h*, D const&>::value, "");
    static_assert(!std::is_constructible<U, A_h*, D&>::value, "");
    static_assert(!std::is_constructible<U, A_h*, D&&>::value, "");

    static_assert(!std::is_constructible<U, B_h*, D const&>::value, "");
    static_assert(!std::is_constructible<U, B_h*, D&>::value, "");
    static_assert(!std::is_constructible<U, B_h*, D&&>::value, "");
  }
  {
    using D = NoCopyMoveDeleter;
    using U = gpu::unique_ptr<A_h[], D&>;
    static_assert(!std::is_constructible<U, A_h*, D const&>::value, "");
    static_assert(std::is_nothrow_constructible<U, A_h*, D&>::value, "");
    static_assert(!std::is_constructible<U, A_h*, D&&>::value, "");
    static_assert(!std::is_constructible<U, A_h*, const D&&>::value, "");

    static_assert(!std::is_constructible<U, B_h*, D const&>::value, "");
    static_assert(!std::is_constructible<U, B_h*, D&>::value, "");
    static_assert(!std::is_constructible<U, B_h*, D&&>::value, "");
    static_assert(!std::is_constructible<U, B_h*, const D&&>::value, "");
  }
  {
    using D = NoCopyMoveDeleter;
    using U = gpu::unique_ptr<A_h[], const D&>;
    static_assert(std::is_nothrow_constructible<U, A_h*, D const&>::value, "");
    static_assert(std::is_nothrow_constructible<U, A_h*, D&>::value, "");
    static_assert(!std::is_constructible<U, A_h*, D&&>::value, "");
    static_assert(!std::is_constructible<U, A_h*, const D&&>::value, "");

    static_assert(!std::is_constructible<U, B_h*, D const&>::value, "");
    static_assert(!std::is_constructible<U, B_h*, D&>::value, "");
    static_assert(!std::is_constructible<U, B_h*, D&&>::value, "");
    static_assert(!std::is_constructible<U, B_h*, const D&&>::value, "");
  }
#endif
}

template <bool IsArray>
TEST_CONSTEXPR_CXX23 void test_basic() {
  typedef typename std::conditional<!IsArray, A_h, A_h[]>::type VT;
  const int expect_alive = IsArray ? 5 : 1;
  { // MoveConstructible deleter (C-1)
    A_h* p = newValue<VT>(expect_alive);
    if (!TEST_IS_CONSTANT_EVALUATED)
      assert(A_h::count == expect_alive);
    gpu::unique_ptr<VT, Deleter<VT> > s(p, Deleter<VT>(5));
    assert(s.get() == p);
    assert(s.get_deleter().state() == 5);
  }
  // In order to get A_h and B_h to be trivially destrictible, we have to use a hack where Deleter is the one
  // decrementing count, but it doesn't quite work for arrays
  if (!TEST_IS_CONSTANT_EVALUATED && !IsArray)
    assert(A_h::count == 0);
  else if (!TEST_IS_CONSTANT_EVALUATED)
    assert((A_h::count -= expect_alive) == 0);
  { // CopyConstructible deleter (C-2)
    A_h* p = newValue<VT>(expect_alive);
    if (!TEST_IS_CONSTANT_EVALUATED)
      assert(A_h::count == expect_alive);
    CopyDeleter<VT> d(5);
    gpu::unique_ptr<VT, CopyDeleter<VT> > s(p, d);
    assert(s.get() == p);
    assert(s.get_deleter().state() == 5);
    d.set_state(6);
    assert(s.get_deleter().state() == 5);
  }
  // In order to get A_h and B_h to be trivially destrictible, we have to use a hack where Deleter is the one
  // decrementing count, but it doesn't quite work for arrays
  if (!TEST_IS_CONSTANT_EVALUATED && !IsArray)
    assert(A_h::count == 0);
  else if (!TEST_IS_CONSTANT_EVALUATED)
    assert((A_h::count -= expect_alive) == 0);
  { // Reference deleter (C-3)
    A_h* p = newValue<VT>(expect_alive);
    if (!TEST_IS_CONSTANT_EVALUATED)
      assert(A_h::count == expect_alive);
    NCDeleter<VT> d(5);
    gpu::unique_ptr<VT, NCDeleter<VT>&> s(p, d);
    assert(s.get() == p);
    assert(&s.get_deleter() == &d);
    assert(s.get_deleter().state() == 5);
    d.set_state(6);
    assert(s.get_deleter().state() == 6);
  }
  // In order to get A_h and B_h to be trivially destrictible, we have to use a hack where Deleter is the one
  // decrementing count, but it doesn't quite work for arrays
  if (!TEST_IS_CONSTANT_EVALUATED && !IsArray)
    assert(A_h::count == 0);
  else if (!TEST_IS_CONSTANT_EVALUATED)
    assert((A_h::count -= expect_alive) == 0);
  { // Const Reference deleter (C-4)
    A_h* p = newValue<VT>(expect_alive);
    if (!TEST_IS_CONSTANT_EVALUATED)
      assert(A_h::count == expect_alive);
    NCConstDeleter<VT> d(5);
    gpu::unique_ptr<VT, NCConstDeleter<VT> const&> s(p, d);
    assert(s.get() == p);
    assert(s.get_deleter().state() == 5);
    assert(&s.get_deleter() == &d);
  }
  // In order to get A_h and B_h to be trivially destrictible, we have to use a hack where Deleter is the one
  // decrementing count, but it doesn't quite work for arrays
  if (!TEST_IS_CONSTANT_EVALUATED && !IsArray)
    assert(A_h::count == 0);
  else if (!TEST_IS_CONSTANT_EVALUATED)
    assert((A_h::count -= expect_alive) == 0);
  if (!TEST_IS_CONSTANT_EVALUATED) { // Void and function pointers (C-6,7)
    typedef typename std::conditional<IsArray, int[], int>::type VT2;
    my_free_called = false;
    {
      int i = 0;
      gpu::unique_ptr<VT2, void (*)(void*)> s(&i, my_free);
      assert(s.get() == &i);
      assert(s.get_deleter() == my_free);
      assert(!my_free_called);
    }
    assert(my_free_called);
  }
}

TEST_CONSTEXPR_CXX23 void test_basic_single() {
  if (!TEST_IS_CONSTANT_EVALUATED) {
    assert(A_h::count == 0);
    assert(B_h::count == 0);
  }
  { // Derived pointers (C-5)
    B_h* p = newValue<B_h>(1);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(A_h::count == 1);
      assert(B_h::count == 1);
    }
    gpu::unique_ptr<A_h, Deleter<A_h> > s(p, Deleter<A_h>(5));
    assert(s.get() == p);
    assert(s.get_deleter().state() == 5);
  }
  // In order to get A_h and B_h to be trivially destrictible, we have to use a hack where Deleter is the one decrementing count
  // But it doesn't work quite right for polymorphic types, so we need to manually decrement B_h::count here 
  if (!TEST_IS_CONSTANT_EVALUATED) {
    assert(A_h::count == 0);
    assert(--B_h::count == 0);

    { // Void and function pointers (C-6,7)
      my_free_called = false;
      {
        int i = 0;
        gpu::unique_ptr<void, void (*)(void*)> s(&i, my_free);
        assert(s.get() == &i);
        assert(s.get_deleter() == my_free);
        assert(!my_free_called);
      }
      assert(my_free_called);
    }
  }
}

template <bool IsArray>
TEST_CONSTEXPR_CXX23 void test_nullptr() {
#if TEST_STD_VER >= 11
  typedef typename std::conditional<!IsArray, A_h, A_h[]>::type VT;
  {
    gpu::unique_ptr<VT, Deleter<VT> > u(nullptr, Deleter<VT>{});
    assert(u.get() == nullptr);
  }
  {
    NCDeleter<VT> d;
    gpu::unique_ptr<VT, NCDeleter<VT>& > u(nullptr, d);
    assert(u.get() == nullptr);
  }
  {
    NCConstDeleter<VT> d;
    gpu::unique_ptr<VT, NCConstDeleter<VT> const& > u(nullptr, d);
    assert(u.get() == nullptr);
  }
#endif
}

TEST_CONSTEXPR_CXX23 bool test() {
  {
    test_basic</*IsArray*/ false>();
    test_nullptr<false>();
    test_basic_single();
    test_sfinae<false>();
    test_noexcept<false>();
  }
  {
    test_basic</*IsArray*/ true>();
    test_nullptr<true>();
    test_sfinae<true>();
    test_sfinae_runtime();
    test_noexcept<true>();
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

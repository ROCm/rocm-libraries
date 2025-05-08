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
// TESTING gpu::unique_ptr::unique_ptr(pointer)
//
// Concerns:
//   1 The pointer constructor works for any default constructible deleter types.
//   2 The pointer constructor accepts pointers to derived types.
//   2 The stored type 'T' is allowed to be incomplete.
//
// Plan
//  1 Construct unique_ptr<T, D>'s with a pointer to 'T' and various deleter
//   types (C-1)
//  2 Construct unique_ptr<T, D>'s with a pointer to 'D' and various deleter
//    types where 'D' is derived from 'T'. (C-1,2)
//  3 Construct a unique_ptr<T, D> with a pointer to 'T' and various deleter
//    types where 'T' is an incomplete type (C-1,3)

// Test unique_ptr(pointer) ctor

#include "gpu/memory"
#include "hip/hip_runtime.h"
#include <cassert>

#include "test_macros.h"
#include "unique_ptr_test_helper.h"

// unique_ptr(pointer) ctor should only require default Deleter ctor

template <bool IsArray>
TEST_CONSTEXPR_CXX23 void test_pointer() {
  typedef typename std::conditional<!IsArray, A_h, A_h[]>::type ValueT;
  const int expect_alive = IsArray ? 5 : 1;
#if TEST_STD_VER >= 11
  {
    using U1 = gpu::unique_ptr_h<ValueT>;
    using U2 = gpu::unique_ptr<ValueT, Deleter<ValueT> >;

    // Test for noexcept
    static_assert(std::is_nothrow_constructible<U1, A_h*>::value, "");
    static_assert(std::is_nothrow_constructible<U2, A_h*>::value, "");

    // Test for explicit
    static_assert(!std::is_convertible<A_h*, U1>::value, "");
    static_assert(!std::is_convertible<A_h*, U2>::value, "");
  }
#endif
  {
    A_h* p = newValue<ValueT>(expect_alive);
    if (!TEST_IS_CONSTANT_EVALUATED)
      assert(A_h::count == expect_alive);

    gpu::unique_ptr_h<ValueT> s(p);
    assert(s.get() == p);
  }
  // In order to get A_h and B_h to be trivially destrictible, we have to use a hack where Deleter is the one
  // decrementing count, but that doesn't work for unique_ptr_h
  if (!TEST_IS_CONSTANT_EVALUATED)
    assert((A_h::count -= expect_alive) == 0);
  {
    A_h* p = newValue<ValueT>(expect_alive);
    if (!TEST_IS_CONSTANT_EVALUATED)
      assert(A_h::count == expect_alive);

    gpu::unique_ptr<ValueT, NCDeleter<ValueT> > s(p);
    assert(s.get() == p);
    assert(s.get_deleter().state() == 0);
  }
  // In order to get A_h and B_h to be trivially destrictible, we have to use a hack where Deleter is the one
  // decrementing count, but it doesn't quite work for arrays
  if (!TEST_IS_CONSTANT_EVALUATED && !IsArray)
    assert(A_h::count == 0);
  else if (!TEST_IS_CONSTANT_EVALUATED)
    assert((A_h::count -= expect_alive) == 0);
  {
    A_h* p = newValue<ValueT>(expect_alive);
    if (!TEST_IS_CONSTANT_EVALUATED)
      assert(A_h::count == expect_alive);

    gpu::unique_ptr<ValueT, DefaultCtorDeleter<ValueT> > s(p);
    assert(s.get() == p);
    assert(s.get_deleter().state() == 0);
  }
  // In order to get A_h and B_h to be trivially destrictible, we have to use a hack where Deleter is the one
  // decrementing count, but it doesn't quite work for arrays
  if (!TEST_IS_CONSTANT_EVALUATED && !IsArray)
    assert(A_h::count == 0);
  else if (!TEST_IS_CONSTANT_EVALUATED)
    assert((A_h::count -= expect_alive) == 0);
}

TEST_CONSTEXPR_CXX23 void test_derived() {
  {
    B_h* p = newValue<B_h>(1);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(A_h::count == 1);
      assert(B_h::count == 1);
    }
    gpu::unique_ptr_h<A_h> s(p);
    assert(s.get() == p);
  }
  // In order to get A_h and B_h to be trivially destrictible, we have to use a hack where Deleter is the one
  // decrementing count, but that doesn't work for unique_ptr_h
  if (!TEST_IS_CONSTANT_EVALUATED) {
    assert(--A_h::count == 0);
    assert(--B_h::count == 0);
  }
  {
    B_h* p = newValue<B_h>(1);
    if (!TEST_IS_CONSTANT_EVALUATED) {
      assert(A_h::count == 1);
      assert(B_h::count == 1);
    }
    gpu::unique_ptr<A_h, NCDeleter<A_h> > s(p);
    assert(s.get() == p);
    assert(s.get_deleter().state() == 0);
  }
  // In order to get A_h and B_h to be trivially destrictible, we have to use a hack where Deleter is the one decrementing count
  // But it doesn't work quite right for polymorphic types, so we need to manually decrement B_h::count here 
  if (!TEST_IS_CONSTANT_EVALUATED) {
    assert(A_h::count == 0);
    assert(--B_h::count == 0);
  }
}

#if TEST_STD_VER >= 11
struct NonDefaultDeleter {
  NonDefaultDeleter() = delete;
  void operator()(void*) const {}
};

struct GenericDeleter {
  void operator()(void*) const;
};
#endif

template <class T>
void TEST_CONSTEXPR_CXX23 test_sfinae() {
#if TEST_STD_VER >= 11
  { // the constructor does not participate in overload resolution when
    // the deleter is a pointer type
    using U = gpu::unique_ptr<T, void (*)(void*)>;
    static_assert(!std::is_constructible<U, T*>::value, "");
  }
  { // the constructor does not participate in overload resolution when
    // the deleter is not default constructible
    using Del = CDeleter<T>;
    using U1 = gpu::unique_ptr<T, NonDefaultDeleter>;
    using U2 = gpu::unique_ptr<T, Del&>;
    using U3 = gpu::unique_ptr<T, Del const&>;
    static_assert(!std::is_constructible<U1, T*>::value, "");
    static_assert(!std::is_constructible<U2, T*>::value, "");
    static_assert(!std::is_constructible<U3, T*>::value, "");
  }
#endif
}

static TEST_CONSTEXPR_CXX23 void test_sfinae_runtime() {
#if TEST_STD_VER >= 11
  { // the constructor does not participate in overload resolution when
    // a base <-> derived conversion would occur.
    using UA = gpu::unique_ptr_h<A_h[]>;
    using UAD = gpu::unique_ptr<A_h[], GenericDeleter>;
    using UAC = gpu::unique_ptr_h<const A_h[]>;
    using UB = gpu::unique_ptr_h<B_h[]>;
    using UBD = gpu::unique_ptr<B_h[], GenericDeleter>;
    using UBC = gpu::unique_ptr_h<const B_h[]>;

    static_assert(!std::is_constructible<UA, B_h*>::value, "");
    static_assert(!std::is_constructible<UB, A_h*>::value, "");
    static_assert(!std::is_constructible<UAD, B_h*>::value, "");
    static_assert(!std::is_constructible<UBD, A_h*>::value, "");
    static_assert(!std::is_constructible<UAC, const B_h*>::value, "");
    static_assert(!std::is_constructible<UBC, const A_h*>::value, "");
  }
#endif
}

// TODO: Are we missing test coverage without this?
// This has been adapted for device-side tests and doesn't work for host tests yet
// DEFINE_AND_RUN_IS_INCOMPLETE_TEST({
//   { doIncompleteTypeTest(1, getNewIncomplete()); }
//   checkNumIncompleteTypeAlive(0);
//   {
//     doIncompleteTypeTest<IncompleteType, NCDeleter<IncompleteType> >(
//         1, getNewIncomplete());
//   }
//   checkNumIncompleteTypeAlive(0);
// })

TEST_CONSTEXPR_CXX23 bool test() {
  {
    test_pointer</*IsArray*/ false>();
    test_derived();
    test_sfinae<int>();
  }
  {
    test_pointer</*IsArray*/ true>();
    test_sfinae<int[]>();
    test_sfinae_runtime();
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

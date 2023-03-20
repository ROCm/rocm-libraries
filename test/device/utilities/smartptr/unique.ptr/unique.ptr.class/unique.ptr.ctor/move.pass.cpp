//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// Test unique_ptr move ctor

#include "gpu/memory"
#include <utility>
#include <cassert>

#include "test_macros.h"
#include "kernel_launcher.h"
#include "unique_ptr_test_helper.h"

//=============================================================================
// TESTING unique_ptr(unique_ptr&&)
//
// Concerns
//   1 The moved from pointer is empty and the new pointer stores the old value.
//   2 The only requirement on the deleter is that it is MoveConstructible
//     or a reference.
//   3 The constructor works for explicitly moved values (i.e. std::move(x))
//   4 The constructor works for true temporaries (e.g. a return value)
//
// Plan
//  1 Explicitly construct unique_ptr<T, D> for various deleter types 'D'.
//    check that the value and deleter have been properly moved. (C-1,2,3)
//
//  2 Use the expression 'sink(source())' to move construct a unique_ptr<T, D>
//    from a temporary. 'source' should return the unique_ptr by value and
//    'sink' should accept the unique_ptr by value. (C-1,2,4)

template <class VT>
__device__ TEST_CONSTEXPR_CXX23 gpu::unique_ptr<VT> source1() {
  return gpu::unique_ptr<VT>(newValue<VT>(1));
}

template <class VT>
__device__ TEST_CONSTEXPR_CXX23 gpu::unique_ptr<VT, Deleter<VT> > source2() {
  return gpu::unique_ptr<VT, Deleter<VT> >(newValue<VT>(1), Deleter<VT>(5));
}

// TODO: Are we missing test coverage without this?
// this isn't valid HIP/CUDA, since dynamic initialization is not supported for __device__ variables
// template <class VT>
// __device__ gpu::unique_ptr<VT, NCDeleter<VT>&> source3() {
//   static __device__ NCDeleter<VT> d(5);
//   return gpu::unique_ptr<VT, NCDeleter<VT>&>(newValue<VT>(1), d);
// }

template <class VT>
__device__ TEST_CONSTEXPR_CXX23 void sink1(gpu::unique_ptr<VT> p) {
  assert(p.get() != nullptr);
}

template <class VT>
__device__ TEST_CONSTEXPR_CXX23 void sink2(gpu::unique_ptr<VT, Deleter<VT> > p) {
  assert(p.get() != nullptr);
  assert(p.get_deleter().state() == 5);
}

// TODO: Are we missing test coverage without this?
// this isn't valid HIP/CUDA, since dynamic initialization is not supported for __device__ variables
// template <class VT>
// __device__ void sink3(gpu::unique_ptr<VT, NCDeleter<VT>&> p) {
//   assert(p.get() != nullptr);
//   assert(p.get_deleter().state() == 5);
//   assert(&p.get_deleter() == &source3<VT>().get_deleter());
// }

template <class ValueT>
__device__ TEST_CONSTEXPR_CXX23 void test_sfinae() {
  typedef gpu::unique_ptr<ValueT> U;
  { // Ensure unique_ptr is non-copyable
    static_assert((!std::is_constructible<U, U const&>::value), "");
    static_assert((!std::is_constructible<U, U&>::value), "");
  }
}

template <bool IsArray>
__device__ TEST_CONSTEXPR_CXX23 void test_basic() {
  typedef typename std::conditional<!IsArray, A, A[]>::type VT;
  const int expect_alive = IsArray ? 5 : 1;
  {
    typedef gpu::unique_ptr<VT> APtr;
    APtr s(newValue<VT>(expect_alive));
    A* p = s.get();
    APtr s2 = std::move(s);
    assert(s2.get() == p);
    assert(s.get() == 0);
    if (!TEST_IS_CONSTANT_EVALUATED)
      assert(A::count == expect_alive);
  }
  if (!TEST_IS_CONSTANT_EVALUATED)
    assert(A::count == 0);
  {
    typedef Deleter<VT> MoveDel;
    typedef gpu::unique_ptr<VT, MoveDel> APtr;
    MoveDel d(5);
    APtr s(newValue<VT>(expect_alive), std::move(d));
    assert(d.state() == 0);
    assert(s.get_deleter().state() == 5);
    A* p = s.get();
    APtr s2 = std::move(s);
    assert(s2.get() == p);
    assert(s.get() == 0);
    if (!TEST_IS_CONSTANT_EVALUATED)
      assert(A::count == expect_alive);
    assert(s2.get_deleter().state() == 5);
    assert(s.get_deleter().state() == 0);
  }
  if (!TEST_IS_CONSTANT_EVALUATED)
    assert(A::count == 0);
  {
    typedef NCDeleter<VT> NonCopyDel;
    typedef gpu::unique_ptr<VT, NonCopyDel&> APtr;

    NonCopyDel d;
    APtr s(newValue<VT>(expect_alive), d);
    A* p = s.get();
    APtr s2 = std::move(s);
    assert(s2.get() == p);
    assert(s.get() == 0);
    if (!TEST_IS_CONSTANT_EVALUATED)
      assert(A::count == expect_alive);
    d.set_state(6);
    assert(s2.get_deleter().state() == d.state());
    assert(s.get_deleter().state() == d.state());
  }
  if (!TEST_IS_CONSTANT_EVALUATED)
    assert(A::count == 0);
  {
    sink1<VT>(source1<VT>());
    if (!TEST_IS_CONSTANT_EVALUATED)
      assert(A::count == 0);
    sink2<VT>(source2<VT>());
    if (!TEST_IS_CONSTANT_EVALUATED)
      assert(A::count == 0);
  }
  if (!TEST_IS_CONSTANT_EVALUATED)
    assert(A::count == 0);
}

template <class VT>
__device__ TEST_CONSTEXPR_CXX23 void test_noexcept() {
#if TEST_STD_VER >= 11
  {
    typedef gpu::unique_ptr<VT> U;
    static_assert(std::is_nothrow_move_constructible<U>::value, "");
  }
  {
    typedef gpu::unique_ptr<VT, Deleter<VT> > U;
    static_assert(std::is_nothrow_move_constructible<U>::value, "");
  }
  {
    typedef gpu::unique_ptr<VT, NCDeleter<VT> &> U;
    static_assert(std::is_nothrow_move_constructible<U>::value, "");
  }
  {
    typedef gpu::unique_ptr<VT, const NCConstDeleter<VT> &> U;
    static_assert(std::is_nothrow_move_constructible<U>::value, "");
  }
#endif
}

__device__ TEST_CONSTEXPR_CXX23 bool test() {
  {
    test_basic</*IsArray*/ false>();
    test_sfinae<int>();
    test_noexcept<int>();
  }
  {
    test_basic</*IsArray*/ true>();
    test_sfinae<int[]>();
    test_noexcept<int[]>();
  }

  return true;
}

// TODO: Are we missing test coverage without this?
// this isn't valid HIP/CUDA, since dynamic initialization is not supported for __device__ variables
// template <bool IsArray>
// __device__ void test_sink3() {
//   typedef typename std::conditional<!IsArray, A, A[]>::type VT;
//   sink3<VT>(source3<VT>());
//   assert(A::count == 0);
// }

__global__ void gmain() {
  // test_sink3</*IsArray*/ false>();
  // test_sink3</*IsArray*/ true>();
  test();
#if TEST_STD_VER >= 23
  static_assert(test());
#endif

}

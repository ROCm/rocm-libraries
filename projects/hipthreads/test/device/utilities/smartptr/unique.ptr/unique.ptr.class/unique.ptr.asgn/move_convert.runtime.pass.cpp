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
#include "kernel_launcher.h"
#include "unique_ptr_test_helper.h"

template <class APtr, class BPtr>
__device__ void testAssign(APtr& aptr, BPtr& bptr) {
  A* p = bptr.get();
  assert(A::count == 2);
  aptr = std::move(bptr);
  assert(aptr.get() == p);
  assert(bptr.get() == 0);
  assert(A::count == 1);
  assert(B::count == 1);
}

template <class LHS, class RHS>
__device__ void checkDeleter(LHS& lhs, RHS& rhs, int LHSState, int RHSState) {
  assert(lhs.get_deleter().state() == LHSState);
  assert(rhs.get_deleter().state() == RHSState);
}

template <class T>
struct NCConvertingDeleter {
  __device__ NCConvertingDeleter() = default;
  __device__ NCConvertingDeleter(NCConvertingDeleter const&) = delete;
  __device__ NCConvertingDeleter(NCConvertingDeleter&&) = default;

  template <class U>
  __device__ NCConvertingDeleter(NCConvertingDeleter<U>&&) {}

  __device__ void operator()(T*) const {}
};

template <class T>
struct NCConvertingDeleter<T[]> {
  __device__ NCConvertingDeleter() = default;
  __device__ NCConvertingDeleter(NCConvertingDeleter const&) = delete;
  __device__ NCConvertingDeleter(NCConvertingDeleter&&) = default;

  template <class U>
  __device__ NCConvertingDeleter(NCConvertingDeleter<U>&&) {}

  __device__ void operator()(T*) const {}
};

struct GenericDeleter {
  void operator()(void*) const;
};

struct NCGenericDeleter {
  __device__ NCGenericDeleter() = default;
  __device__ NCGenericDeleter(NCGenericDeleter const&) = delete;
  __device__ NCGenericDeleter(NCGenericDeleter&&) = default;

  __device__ void operator()(void*) const {}
};

__device__ void test_sfinae() {
  using DA = NCConvertingDeleter<A[]>;        // non-copyable deleters
  using DAC = NCConvertingDeleter<const A[]>; // non-copyable deleters

  using UA = gpu::unique_ptr<A[]>;
  using UAC = gpu::unique_ptr<const A[]>;
  using UAD = gpu::unique_ptr<A[], DA>;
  using UACD = gpu::unique_ptr<const A[], DAC>;

  { // cannot move from an lvalue
    static_assert(std::is_assignable<UAC, UA&&>::value, "");
    static_assert(!std::is_assignable<UAC, UA&>::value, "");
    static_assert(!std::is_assignable<UAC, const UA&>::value, "");
  }
  { // cannot move if the deleter-types cannot convert
    static_assert(std::is_assignable<UACD, UAD&&>::value, "");
    static_assert(!std::is_assignable<UACD, UAC&&>::value, "");
    static_assert(!std::is_assignable<UAC, UACD&&>::value, "");
  }
  { // cannot move-convert with reference deleters of different types
    using UA1 = gpu::unique_ptr<A[], DA&>;
    using UA2 = gpu::unique_ptr<A[], DAC&>;
    static_assert(!std::is_assignable<UA1, UA2&&>::value, "");
  }
  { // cannot move-convert with reference deleters of different types
    using UA1 = gpu::unique_ptr<A[], const DA&>;
    using UA2 = gpu::unique_ptr<A[], const DAC&>;
    static_assert(!std::is_assignable<UA1, UA2&&>::value, "");
  }
  { // cannot move-convert from unique_ptr<Single>
    using UA1 = gpu::unique_ptr<A[]>;
    using UA2 = gpu::unique_ptr<A>;
    static_assert(!std::is_assignable<UA1, UA2&&>::value, "");
  }
  { // cannot move-convert from unique_ptr<Array[]>
    using UA1 = gpu::unique_ptr<A[], NCGenericDeleter>;
    using UA2 = gpu::unique_ptr<A, NCGenericDeleter>;
    static_assert(!std::is_assignable<UA1, UA2&&>::value, "");
  }
}

__global__ void gmain() {
  test_sfinae();
  // FIXME: add tests

}

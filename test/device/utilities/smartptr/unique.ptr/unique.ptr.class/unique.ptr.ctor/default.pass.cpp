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
// TESTING gpu::unique_ptr::unique_ptr()
//
// Concerns:
//   1 The default constructor works for any default constructible deleter types.
//   2 The stored type 'T' is allowed to be incomplete.
//
// Plan
//  1 Default construct unique_ptr's with various deleter types (C-1)
//  2 Default construct a unique_ptr with an incomplete element_type and
//    various deleter types (C-1,2)

#include "gpu/memory"
#include <cassert>
#include "test_macros.h"
#include "kernel_launcher.h"

#include "test_macros.h"
#include "kernel_launcher.h"
#include "deleter_types.h"
#include "unique_ptr_test_helper.h"

#if TEST_STD_VER >= 11
// TODO: Are we missing test coverage without this?
// this isn't valid HIP/CUDA, since dynamic initialization is not supported for __device__ variables
// __device__ TEST_CONSTINIT gpu::unique_ptr<int> global_static_unique_ptr_single;
// __device__ TEST_CONSTINIT gpu::unique_ptr<int[]> global_static_unique_ptr_runtime;

struct NonDefaultDeleter {
  __device__ NonDefaultDeleter() = delete;
  __device__ TEST_CONSTEXPR_CXX23 void operator()(void*) const {}
};
#endif

template <class ElemType>
__device__ TEST_CONSTEXPR_CXX23 void test_sfinae() {
#if TEST_STD_VER >= 11
  { // the constructor does not participate in overload resolution when
    // the deleter is a pointer type
    using U = gpu::unique_ptr<ElemType, void (*)(void*)>;
    static_assert(!std::is_default_constructible<U>::value, "");
  }
  { // the constructor does not participate in overload resolution when
    // the deleter is not default constructible
    using Del = CDeleter<ElemType>;
    using U1  = gpu::unique_ptr<ElemType, NonDefaultDeleter>;
    using U2  = gpu::unique_ptr<ElemType, Del&>;
    using U3  = gpu::unique_ptr<ElemType, Del const&>;
    static_assert(!std::is_default_constructible<U1>::value, "");
    static_assert(!std::is_default_constructible<U2>::value, "");
    static_assert(!std::is_default_constructible<U3>::value, "");
  }
#endif
}

template <class ElemType>
__device__ TEST_CONSTEXPR_CXX23 bool test_basic() {
#if TEST_STD_VER >= 11
  {
    using U1 = gpu::unique_ptr<ElemType>;
    using U2 = gpu::unique_ptr<ElemType, Deleter<ElemType> >;
    static_assert(std::is_nothrow_default_constructible<U1>::value, "");
    static_assert(std::is_nothrow_default_constructible<U2>::value, "");
  }
#endif
  {
    gpu::unique_ptr<ElemType> p;
    assert(p.get() == 0);
  }
  {
    gpu::unique_ptr<ElemType, NCDeleter<ElemType> > p;
    assert(p.get() == 0);
    assert(p.get_deleter().state() == 0);
    p.get_deleter().set_state(5);
    assert(p.get_deleter().state() == 5);
  }
  {
    gpu::unique_ptr<ElemType, DefaultCtorDeleter<ElemType> > p;
    assert(p.get() == 0);
    assert(p.get_deleter().state() == 0);
  }

  return true;
}

DEFINE_AND_RUN_IS_INCOMPLETE_TEST({
  doIncompleteTypeTest(0);
  doIncompleteTypeTest<IncompleteType, Deleter<IncompleteType> >(0);
} {
  doIncompleteTypeTest<IncompleteType[]>(0);
  doIncompleteTypeTest<IncompleteType[], Deleter<IncompleteType[]> >(0);
})

__device__ TEST_CONSTEXPR_CXX23 bool test() {
  {
    test_sfinae<int>();
    test_basic<int>();
  }
  {
    test_sfinae<int[]>();
    test_basic<int[]>();
  }

  return true;
}

__global__ void gmain() {
  test();
#if TEST_STD_VER >= 23
  static_assert(test());
#endif

}

//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// <memory>

// unique_ptr

//=============================================================================
// TESTING cuda::std::unique_ptr::unique_ptr()
//
// Concerns:
//   1 The default constructor works for any default constructible deleter types.
//   2 The stored type 'T' is allowed to be incomplete.
//
// Plan
//  1 Default construct unique_ptr's with various deleter types (C-1)
//  2 Default construct a unique_ptr with an incomplete element_type and
//    various deleter types (C-1,2)

#include <cuda/std/__memory_>
#include <cuda/std/cassert>

#include "deleter_types.h"
#include "test_macros.h"
#include "unique_ptr_test_helper.h"

#if !TEST_COMPILER(NVRTC) && !defined(TEST_COMPILER_HIPRTC) // no dynamic initialization
_LIBCUDACXX_SAFE_STATIC cuda::std::unique_ptr<int> global_static_unique_ptr_single;
_LIBCUDACXX_SAFE_STATIC cuda::std::unique_ptr<int[]> global_static_unique_ptr_runtime;
#endif // TEST_COMPILER(NVRTC)

struct NonDefaultDeleter
{
  NonDefaultDeleter() = delete;
  __host__ __device__ TEST_CONSTEXPR_CXX23 void operator()(void*) const {}
};

template <class ElemType>
__host__ __device__ TEST_CONSTEXPR_CXX23 void test_sfinae()
{
  { // the constructor does not participate in overload resolution when
    // the deleter is a pointer type
    using U = cuda::std::unique_ptr<ElemType, void (*)(void*)>;
    static_assert(!cuda::std::is_default_constructible<U>::value, "");
  }
  { // the constructor does not participate in overload resolution when
    // the deleter is not default constructible
    using Del = CDeleter<ElemType>;
    using U1  = cuda::std::unique_ptr<ElemType, NonDefaultDeleter>;
    using U2  = cuda::std::unique_ptr<ElemType, Del&>;
    using U3  = cuda::std::unique_ptr<ElemType, Del const&>;
    static_assert(!cuda::std::is_default_constructible<U1>::value, "");
    static_assert(!cuda::std::is_default_constructible<U2>::value, "");
    static_assert(!cuda::std::is_default_constructible<U3>::value, "");
  }
}

template <class ElemType>
__host__ __device__ TEST_CONSTEXPR_CXX23 bool test_basic()
{
  {
    using U1 = cuda::std::unique_ptr<ElemType>;
    using U2 = cuda::std::unique_ptr<ElemType, Deleter<ElemType>>;
    static_assert(cuda::std::is_nothrow_default_constructible<U1>::value, "");
    static_assert(cuda::std::is_nothrow_default_constructible<U2>::value, "");
  }
  {
    cuda::std::unique_ptr<ElemType> p;
    assert(p.get() == 0);
  }
  {
    cuda::std::unique_ptr<ElemType, NCDeleter<ElemType>> p;
    assert(p.get() == 0);
    assert(p.get_deleter().state() == 0);
    p.get_deleter().set_state(5);
    assert(p.get_deleter().state() == 5);
  }
  {
    cuda::std::unique_ptr<ElemType, DefaultCtorDeleter<ElemType>> p;
    assert(p.get() == 0);
    assert(p.get_deleter().state() == 0);
  }

  return true;
}

#ifndef __CUDACC__
DEFINE_AND_RUN_IS_INCOMPLETE_TEST(
  {
    doIncompleteTypeTest(0);
    doIncompleteTypeTest<IncompleteType, Deleter<IncompleteType>>(0);
  } {
    doIncompleteTypeTest<IncompleteType[]>(0);
    doIncompleteTypeTest<IncompleteType[], Deleter<IncompleteType[]>>(0);
  })
#endif // __CUDACC__

__host__ __device__ TEST_CONSTEXPR_CXX23 bool test()
{
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

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2023
  static_assert(test());
#endif // TEST_STD_VER >= 2023

  return 0;
}

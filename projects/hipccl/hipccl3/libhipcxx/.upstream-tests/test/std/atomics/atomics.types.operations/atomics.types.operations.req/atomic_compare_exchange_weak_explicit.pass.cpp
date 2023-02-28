//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

//
// UNSUPPORTED: hipcc
// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70
//  ... assertion fails line 38

// <cuda/std/atomic>

// template <class T>
//     bool
//     atomic_compare_exchange_weak_explicit(volatile atomic<T>* obj, T* expc,
//                                           T desr,
//                                           memory_order s, memory_order f);
//
// template <class T>
//     bool
//     atomic_compare_exchange_weak_explicit(atomic<T>* obj, T* expc, T desr,
//                                           memory_order s, memory_order f);

#include <cuda/std/atomic>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "atomic_helpers.h"
#include "cuda_space_selector.h"
#include "test_macros.h"
#include <cmpxchg_loop.h>

template <class T, template <typename, typename> typename Selector, cuda::thread_scope>
struct TestFn
{
  __host__ __device__ void operator()() const
  {
    {
      typedef cuda::std::atomic<T> A;
      Selector<A, constructor_initializer> sel;
      A& a = *sel.construct();
      T t(T(1));
      cuda::std::atomic_init(&a, t);
      assert(c_cmpxchg_weak_loop(&a, &t, T(2), cuda::std::memory_order_seq_cst, cuda::std::memory_order_seq_cst)
             == true);
      assert(a == T(2));
      assert(t == T(1));
      assert(cuda::std::atomic_compare_exchange_weak_explicit(
               &a, &t, T(3), cuda::std::memory_order_seq_cst, cuda::std::memory_order_seq_cst)
             == false);
      assert(a == T(2));
      assert(t == T(2));
    }
    {
      typedef cuda::std::atomic<T> A;
      Selector<volatile A, constructor_initializer> sel;
      volatile A& a = *sel.construct();
      T t(T(1));
      cuda::std::atomic_init(&a, t);
      assert(c_cmpxchg_weak_loop(&a, &t, T(2), cuda::std::memory_order_seq_cst, cuda::std::memory_order_seq_cst)
             == true);
      assert(a == T(2));
      assert(t == T(1));
      assert(cuda::std::atomic_compare_exchange_weak_explicit(
               &a, &t, T(3), cuda::std::memory_order_seq_cst, cuda::std::memory_order_seq_cst)
             == false);
      assert(a == T(2));
      assert(t == T(2));
    }
  }
};

int main(int, char**)
{
  NV_DISPATCH_TARGET(NV_IS_HOST,
                     (TestEachAtomicType<TestFn, local_memory_selector>()();),
                     NV_PROVIDES_SM_70,
                     (TestEachAtomicType<TestFn, local_memory_selector>()();))

  NV_IF_TARGET(
    NV_IS_DEVICE,
    (TestEachAtomicType<TestFn, shared_memory_selector>()(); TestEachAtomicType<TestFn, global_memory_selector>()();))

  return 0;
}

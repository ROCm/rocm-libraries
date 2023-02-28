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
//  ... fails assertion line 31

// <cuda/std/atomic>

// template <class T>
//     T
//     atomic_exchange(volatile atomic<T>* obj, T desr);
//
// template <class T>
//     T
//     atomic_exchange(atomic<T>* obj, T desr);

#include <cuda/std/atomic>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "atomic_helpers.h"
#include "cuda_space_selector.h"
#include "test_macros.h"

template <class T, template <typename, typename> typename Selector, cuda::thread_scope>
struct TestFn
{
  __host__ __device__ void operator()() const
  {
    typedef cuda::std::atomic<T> A;
    Selector<A, constructor_initializer> sel;
    A& t = *sel.construct();
    cuda::std::atomic_init(&t, T(1));
    assert(cuda::std::atomic_exchange(&t, T(2)) == T(1));
    assert(t == T(2));
    Selector<volatile A, constructor_initializer> vsel;
    volatile A& vt = *vsel.construct();
    cuda::std::atomic_init(&vt, T(3));
    assert(cuda::std::atomic_exchange(&vt, T(4)) == T(3));
    assert(vt == T(4));
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

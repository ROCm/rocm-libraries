//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
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
// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

// <cuda/atomic>

#include <cuda/atomic>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#ifdef __HIP__
#include <hip/hip_fp16.h>
#endif

#include "atomic_helpers.h"
#include "cuda_space_selector.h"
#include "test_macros.h"

template <class T, template <typename, typename> class Selector, cuda::thread_scope>
struct TestFn
{
  __host__ __device__ void operator()() const
  {
    {
      typedef cuda::atomic<T> A;
      Selector<A, constructor_initializer> sel;
      A& t = *sel.construct();
      t.fetch_min(4);
    }
    {
      typedef cuda::atomic<T> A;
      Selector<volatile A, constructor_initializer> sel;
      volatile A& t = *sel.construct();
      t.fetch_max(4);
    }
    T tmp = T(0);
    {
      cuda::atomic_ref<T> t(tmp);
      t.fetch_min(4);
    }
    {
      cuda::atomic_ref<T> t(tmp);
      t.fetch_max(4);
    }
  }
};

int main(int, char**)
{
  NV_DISPATCH_TARGET(
    NV_IS_HOST, TestFn<__half, local_memory_selector, cuda::thread_scope::thread_scope_thread>()();
    , NV_PROVIDES_SM_70, (TestFn<__half, local_memory_selector, cuda::thread_scope::thread_scope_thread>()();))

  NV_IF_TARGET(NV_IS_DEVICE,
               (TestFn<__half, shared_memory_selector, cuda::thread_scope::thread_scope_thread>()();
                TestFn<__half, global_memory_selector, cuda::thread_scope::thread_scope_thread>()();))

  return 0;
}

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
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++98, c++03
// UNSUPPORTED: pre-sm-70
// UNSUPPORTED: hipcc

// <cuda/std/atomic>

#include <cuda/std/atomic>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "../atomics.types.operations.req/atomic_helpers.h"
#include "concurrent_agents.h"
#include "cuda_space_selector.h"
#include "test_macros.h"

template <class T, template <typename, typename> typename Selector, cuda::thread_scope Scope>
struct TestFn
{
  __host__ __device__ void operator()() const
  {
    typedef cuda::std::atomic_ref<T> A;

    SHARED T* t;
    execute_on_main_thread([&] {
      t = (T*) malloc(sizeof(A));
      A a(*t);
      a.store(T(1));
      assert(a.load() == T(1));
      a.wait(T(0));
    });

    {
      A a(*t);

      auto agent_notify = LAMBDA()
      {
        a.store(T(3));
        a.notify_one();
      };

      auto agent_wait = LAMBDA()
      {
        a.wait(T(1));
      };

      concurrent_agents_launch(agent_notify, agent_wait);
    }
  }
};

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, cuda_thread_count = 2;)

  TestEachAtomicRefType<TestFn, shared_memory_selector>()();

  return 0;
}

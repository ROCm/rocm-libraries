//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
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

// UNSUPPORTED: nvrtc, hiprtc, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

#include <cuda/std/atomic>
#include <cuda/std/cassert>

#include "common.h"

__host__ __device__ void validate_not_lock_free()
{
  cuda::std::atomic<big_not_lockfree_type> test;
  assert(!test.is_lock_free());
}

void kernel_invoker()
{
  validate_pinned<cuda::std::atomic<signed char>, arithmetic_atomic_testers>();
  validate_pinned<cuda::std::atomic<signed short>, arithmetic_atomic_testers>();
  validate_pinned<cuda::std::atomic<signed int>, arithmetic_atomic_testers>();
  validate_pinned<cuda::std::atomic<signed long>, arithmetic_atomic_testers>();
  validate_pinned<cuda::std::atomic<signed long long>, arithmetic_atomic_testers>();

  validate_pinned<cuda::std::atomic<unsigned char>, bitwise_atomic_testers>();
  validate_pinned<cuda::std::atomic<unsigned short>, bitwise_atomic_testers>();
  validate_pinned<cuda::std::atomic<unsigned int>, bitwise_atomic_testers>();
  validate_pinned<cuda::std::atomic<unsigned long>, bitwise_atomic_testers>();
  validate_pinned<cuda::std::atomic<unsigned long long>, bitwise_atomic_testers>();

  validate_pinned<cuda::std::atomic<float>, arithmetic_atomic_testers>();
  validate_pinned<cuda::std::atomic<double>, arithmetic_atomic_testers>();

  validate_pinned<cuda::std::atomic<big_not_lockfree_type>, basic_testers>();
}

int main(int arg, char** argv)
{
  validate_not_lock_free();

  NV_IF_TARGET(NV_IS_HOST, (kernel_invoker();))

  return 0;
}

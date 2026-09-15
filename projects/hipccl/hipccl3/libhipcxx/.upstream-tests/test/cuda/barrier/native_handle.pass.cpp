//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
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

// NOTE(HIP/AMD): currently barrier is not supported on AMD hardware
// UNSUPPORTED: hipcc, hiprtc

// UNSUPPORTED: pre-sm-80

#include <cuda/barrier>

#include "cuda_space_selector.h"
#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(static_var_with_dynamic_init)
TEST_NV_DIAG_SUPPRESS(set_but_not_used)

__device__ void test()
{
  __shared__ cuda::barrier<cuda::thread_scope_block>* b;
  shared_memory_selector<cuda::barrier<cuda::thread_scope_block>, constructor_initializer> sel;
  b = sel.construct(2);

  uint64_t token;
  asm volatile("mbarrier.arrive.b64 %0, [%1];" : "=l"(token) : "l"(cuda::device::barrier_native_handle(*b)) : "memory");
  (void) token;

  b->arrive_and_wait();
}

int main(int argc, char** argv)
{
  NV_IF_TARGET(NV_PROVIDES_SM_80, test();)

  return 0;
}

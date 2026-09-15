//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
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

// UNSUPPORTED: pre-sm-70

// NOTE(HIP/AMD): currently barrier is not supported on AMD hardware
// UNSUPPORTED: hipcc, hiprtc

#include <cuda/barrier>

#include "cuda_space_selector.h"

template <cuda::thread_scope Sco, template <typename, typename> class BarrierSelector>
__host__ __device__ void test()
{
  cuda::barrier<Sco> b(3);

  init(&b, 2);

  auto token = b.arrive();
  b.arrive_and_wait();
  b.wait(std::move(token));
}

template <cuda::thread_scope Sco>
__host__ __device__ void test_select_barrier()
{
  test<Sco, local_memory_selector>();
  NV_IF_TARGET(NV_IS_DEVICE, (test<Sco, shared_memory_selector>(); test<Sco, global_memory_selector>();))
}

int main(int argc, char** argv)
{
  test_select_barrier<cuda::thread_scope_system>();
  test_select_barrier<cuda::thread_scope_device>();
  test_select_barrier<cuda::thread_scope_block>();
  test_select_barrier<cuda::thread_scope_thread>();

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// MIT License
//
// Modifications Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// UNSUPPORTED: pre-sm-70
// UNSUPPORTED: !nvcc
// UNSUPPORTED: nvrtc, hiprtc

#include "utils.h"
#include <cooperative_groups.h>

// TODO: global-shared
// TODO: read  const
__device__ __host__ __noinline__ void test_memcpy_async()
{
  size_t ARR_SZ = 1 << 10;
  int* arr0     = nullptr;
  int* arr1     = nullptr;
  cuda::access_property ap(cuda::access_property::persisting{});
  cuda::barrier<cuda::thread_scope_system> bar0, bar1, bar2, bar3;
  init(&bar0, 1);
  init(&bar1, 1);
  init(&bar2, 1);
  init(&bar3, 1);

  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (arr0 = (int*) malloc(ARR_SZ * sizeof(int)); arr1 = (int*) malloc(ARR_SZ * sizeof(int));),
    (assert_rt(cudaMallocManaged((void**) &arr0, ARR_SZ * sizeof(int)));
     assert_rt(cudaMallocManaged((void**) &arr1, ARR_SZ * sizeof(int)));
     assert_rt(cudaDeviceSynchronize());))

  cuda::annotated_ptr<int, cuda::access_property> ann0{arr0, ap};
  cuda::annotated_ptr<int, cuda::access_property> ann1{arr1, ap};
  // cuda::annotated_ptr<const int, cuda::access_property> cann0{arr0, ap};

  for (size_t i = 0; i < ARR_SZ; ++i)
  {
    arr0[i] = static_cast<int>(i);
    arr1[i] = 0;
  }

  cuda::memcpy_async(ann1, ann0, ARR_SZ * sizeof(int), bar0);
  // cuda::memcpy_async(ann1, cann0, ARR_SZ * sizeof(int), bar0);
  bar0.arrive_and_wait();

  for (size_t i = 0; i < ARR_SZ; ++i)
  {
    if (arr1[i] != (int) i)
    {
      DPRINTF(stderr, "%p:&arr1[i] == %d, should be:%lu\n", &arr1[i], arr1[i], i);
      assert(arr1[i] == static_cast<int>(i));
    }

    arr1[i] = 0;
  }

  cuda::memcpy_async(arr1, ann0, ARR_SZ * sizeof(int), bar1);
  // cuda::memcpy_async(arr1, cann0, ARR_SZ * sizeof(int), bar1);
  bar1.arrive_and_wait();

  for (size_t i = 0; i < ARR_SZ; ++i)
  {
    if (arr1[i] != (int) i)
    {
      DPRINTF(stderr, "%p:&arr1[i] == %d, should be:%lu\n", &arr1[i], arr1[i], i);
      assert(arr1[i] == static_cast<int>(i));
    }

    arr1[i] = 0;
  }

  NV_IF_TARGET(
    NV_IS_DEVICE,
    (
      auto group = cooperative_groups::this_thread_block();

      cuda::memcpy_async(group, ann1, ann0, ARR_SZ * sizeof(int), bar2);
      // cuda::memcpy_async(group, ann1, cann0, ARR_SZ * sizeof(int), bar2);
      bar2.arrive_and_wait();

      for (size_t i = 0; i < ARR_SZ; ++i) {
        if (arr1[i] != (int) i)
        {
          DPRINTF(stderr, "%p:&arr1[i] == %d, should be:%lu\n", &arr1[i], arr1[i], i);
          assert(arr1[i] == i);
        }

        arr1[i] = 0;
      }

      cuda::memcpy_async(group, arr1, ann0, ARR_SZ * sizeof(int), bar3);
      // cuda::memcpy_async(group, arr1, cann0, ARR_SZ * sizeof(int), bar3);
      bar3.arrive_and_wait();

      for (size_t i = 0; i < ARR_SZ; ++i) {
        if (arr1[i] != (int) i)
        {
          DPRINTF(stderr, "%p:&arr1[i] == %d, should be:%lu\n", &arr1[i], arr1[i], i);
          assert(arr1[i] == i);
        }

        arr1[i] = 0;
      }))

  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (free(arr0); free(arr1);), (assert_rt(cudaFree(arr0)); assert_rt(cudaFree(arr1));))
}

int main(int argc, char** argv)
{
  test_memcpy_async();
  return 0;
}

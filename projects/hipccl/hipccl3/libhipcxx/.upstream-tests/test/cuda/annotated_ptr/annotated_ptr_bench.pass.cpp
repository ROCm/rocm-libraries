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

__device__ void annotated_ptr_timing_dev(int* in, int* out)
{
  cuda::access_property ap(cuda::access_property::persisting{});
  // Retrieve global id
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  cuda::annotated_ptr<int, cuda::access_property> in_ann{in, ap};
  cuda::annotated_ptr<int, cuda::access_property> out_ann{out, ap};

  DPRINTF("&out[i]:%p = &in[i]:%p for i = %d\n", &out[i], &in[i], i);
  DPRINTF("&out[i]:%p = &in_ann[i]:%p for i = %d\n", &out_ann[i], &in_ann[i], i);

  out_ann[i] = in_ann[i];
};

__global__ void annotated_ptr_timing(int* in, int* out)
{
  annotated_ptr_timing_dev(in, out);
}

__device__ void ptr_timing_dev(int* in, int* out)
{
  // Retrieve global id
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  DPRINTF("&out[i]:%p = &in[i]:%p for i = %d\n", &out[i], &in[i], i);
  out[i] = in[i];
};

__global__ void ptr_timing(int* in, int* out)
{
  ptr_timing_dev(in, out);
};

__device__ __host__ __noinline__ void bench()
{
#ifndef __CUDA_ARCH__
  static const size_t ARR_SZ     = 1 << 22;
  static const size_t THREAD_CNT = 128;
  static const size_t BLOCK_CNT  = ARR_SZ / THREAD_CNT;
  const dim3 threads(THREAD_CNT, 1, 1), blocks(BLOCK_CNT, 1, 1);
  cudaEvent_t start, stop;
#else
  static const size_t ARR_SZ = 1 << 10;
#endif
  int* arr0            = nullptr;
  int* arr1            = nullptr;
  float annotated_time = 0.f, pointer_time = 0.f;

#ifdef __CUDA_ARCH__
  arr0 = (int*) malloc(ARR_SZ * sizeof(int));
  arr1 = (int*) malloc(ARR_SZ * sizeof(int));
#else
  assert_rt(cudaMallocManaged((void**) &arr0, ARR_SZ * sizeof(int)));
  assert_rt(cudaMallocManaged((void**) &arr1, ARR_SZ * sizeof(int)));
  assert_rt(cudaDeviceSynchronize());
#endif

#ifdef __CUDA_ARCH__
  ptr_timing_dev(arr0, arr1);
#else
  ptr_timing<<<blocks, threads>>>(arr0, arr1);
  assert_rt(cudaDeviceSynchronize());
#endif

  for (size_t i = 0; i < ARR_SZ; ++i)
  {
    arr0[i] = static_cast<int>(i);
    arr1[i] = 0;
  }

#ifdef __CUDA_ARCH__
  ptr_timing_dev(arr0, arr1);
#else
  assert_rt(cudaDeviceSynchronize());
  assert_rt(cudaEventCreate(&start));
  assert_rt(cudaEventCreate(&stop));
  assert_rt(cudaEventRecord(start));
  ptr_timing<<<blocks, threads>>>(arr0, arr1);
  assert_rt(cudaEventRecord(stop));
  assert_rt(cudaEventSynchronize(stop));
  assert_rt(cudaEventElapsedTime(&pointer_time, start, stop));
  assert_rt(cudaEventDestroy(start));
  assert_rt(cudaEventDestroy(stop));
  assert_rt(cudaDeviceSynchronize());

  for (size_t i = 0; i < ARR_SZ; ++i)
  {
    if (arr1[i] != (int) i)
    {
      DPRINTF("arr1[%d] == %d, should be:%d\n", i, arr1[i], i);
      assert(arr1[i] == static_cast<int>(i));
    }

    arr1[i] = 0;
  }
#endif

  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
                    (annotated_ptr_timing_dev(arr0, arr1);),
                    (assert_rt(cudaDeviceSynchronize()); annotated_ptr_timing<<<blocks, threads>>>(arr0, arr1);
                     assert_rt(cudaDeviceSynchronize());))

  for (size_t i = 0; i < ARR_SZ; ++i)
  {
    arr0[i] = static_cast<int>(i);
    arr1[i] = 0;
  }

  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (annotated_ptr_timing_dev(arr0, arr1);),
    (assert_rt(cudaDeviceSynchronize()); assert_rt(cudaEventCreate(&start)); assert_rt(cudaEventCreate(&stop));
     assert_rt(cudaEventRecord(start));
     annotated_ptr_timing<<<blocks, threads>>>(arr0, arr1);
     assert_rt(cudaEventRecord(stop));
     assert_rt(cudaEventSynchronize(stop));
     assert_rt(cudaEventElapsedTime(&annotated_time, start, stop));
     assert_rt(cudaEventDestroy(start));
     assert_rt(cudaEventDestroy(stop));
     assert_rt(cudaDeviceSynchronize());

     for (size_t i = 0; i < ARR_SZ; ++i) {
       if (arr1[i] != (int) i)
       {
         DPRINTF("arr1[%d] == %d, should be:%d\n", i, arr1[i], i);
         assert(arr1[i] == static_cast<int>(i));
       }

       arr1[i] = 0;
     }))

  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (free(arr0); free(arr1);), (assert_rt(cudaFree(arr0)); assert_rt(cudaFree(arr1));))

  printf("array(ms):%f, arrotated_ptr(ms):%f\n", pointer_time, annotated_time);
}

int main(int argc, char** argv)
{
  bench();
  return 0;
}

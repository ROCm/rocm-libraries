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

#include "test_macros.h"
#if defined(TEST_COMPILER_MSVC)
#  pragma warning(disable : 4505)
#endif

#include <cuda/annotated_ptr>
#include <cuda/std/cassert>

#if defined(DEBUG)
#  define DPRINTF(...)     \
    {                      \
      printf(__VA_ARGS__); \
    }
#else
#  define DPRINTF(...) \
    do                 \
    {                  \
    } while (false)
#endif

__device__ __host__ void assert_rt_wrap(hipError_t code, const char* file, int line)
{
  if (code != hipSuccess)
  {
#ifndef TEST_COMPILER_NVRTC
    printf("assert: %s %s %d\n", GetErrorString(code), file, line);
#endif
    assert(code == hipSuccess);
  }
}
#define assert_rt(ret)                         \
  {                                            \
    assert_rt_wrap((ret), __FILE__, __LINE__); \
  }

template <typename T, int N>
__device__ __host__ __noinline__ T* alloc(bool shared = false)
{
  T* arr = nullptr;

  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (
      if (!shared) { arr = (T*) malloc(N * sizeof(T)); } else {
        __shared__ T data[N];
        arr = data;
      }),
    assert_rt(hipMallocManaged((void**) &arr, N * sizeof(T)));)

  for (int i = 0; i < N; ++i)
  {
    arr[i] = i;
  }
  return arr;
}

template <typename T>
__device__ __host__ __noinline__ void dealloc(T* arr, bool shared)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (if (!shared) free(arr);), assert_rt(hipFree(arr));)
}

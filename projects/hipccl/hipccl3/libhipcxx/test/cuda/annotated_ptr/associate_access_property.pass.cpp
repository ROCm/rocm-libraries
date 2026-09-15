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
// UNSUPPORTED: !nvcc
// UNSUPPORTED: nvrtc, hiprtc

#include "utils.h"
#define ARR_SZ 128

template <typename T, typename P>
__device__ __host__ __noinline__ void test(P ap)
{
  T* arr = global_alloc<T, ARR_SZ>();

  arr = cuda::associate_access_property(arr, ap);

  for (int i = 0; i < ARR_SZ; ++i)
  {
    assert(arr[i] == i);
  }

  dealloc<T>(arr);
}

template <typename T, typename P>
__device__ __host__ __noinline__ void test_shared(P ap)
{
  T* arr = shared_alloc<T, ARR_SZ>();

  arr = cuda::associate_access_property(arr, ap);

  for (int i = 0; i < ARR_SZ; ++i)
  {
    assert(arr[i] == i);
  }
}

__device__ __host__ __noinline__ void test_all()
{
  test<int>(cuda::access_property::normal{});
  test<int>(cuda::access_property::persisting{});
  test<int>(cuda::access_property::streaming{});
  test<int>(cuda::access_property::global{});
  test<int>(cuda::access_property{});
  NV_IF_TARGET(NV_IS_DEVICE, (test_shared<int>(cuda::access_property::shared{});))
}

int main(int argc, char** argv)
{
  test_all();
  return 0;
}

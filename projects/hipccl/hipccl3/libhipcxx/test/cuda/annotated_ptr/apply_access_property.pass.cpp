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

constexpr size_t array_size = 128;

template <typename T, typename P>
__device__ __host__ __noinline__ void test(P ap)
{
  T* arr = global_alloc<T, array_size>();

  cuda::apply_access_property(arr, array_size * sizeof(T), ap);

  for (size_t i = 0; i < array_size; ++i)
  {
    assert(static_cast<size_t>(arr[i]) == i);
  }

  dealloc<T>(arr);
}

template <typename T, typename P>
__device__ __host__ __noinline__ void test_aligned(P ap)
{
  T* arr = global_alloc<T, array_size>();

  cuda::apply_access_property(arr, cuda::aligned_size_t<sizeof(T)>(array_size * sizeof(T)), ap);

  for (size_t i = 0; i < array_size; ++i)
  {
    assert(static_cast<size_t>(arr[i]) == i);
  }

  dealloc<T>(arr);
}

__device__ __host__ __noinline__ void test_all()
{
  test<int>(cuda::access_property::normal{});
  test<int>(cuda::access_property::persisting{});
  test_aligned<int>(cuda::access_property::normal{});
  test_aligned<int>(cuda::access_property::persisting{});
}

int main(int argc, char** argv)
{
  test_all();
  return 0;
}

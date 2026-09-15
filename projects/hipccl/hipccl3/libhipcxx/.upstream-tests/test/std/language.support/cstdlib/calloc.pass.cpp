//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// MIT License
//
// Modifications Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/cstdlib>
#include <cuda/std/limits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_calloc_success(cuda::std::size_t n)
{
  T* ptr = static_cast<T*>(cuda::std::calloc(n, sizeof(T)));

  // check that the memory was allocated
  assert(ptr != nullptr);

  // check that the memory is zeroed
  for (cuda::std::size_t i = 0; i < n; ++i)
  {
    assert(ptr[i] == T{});
  }

  // check memory alignment
  assert(((alignof(T) - 1) & reinterpret_cast<cuda::std::uintptr_t>(ptr)) == 0);

  cuda::std::free(ptr);
}

template <class T>
__host__ __device__ void test_calloc_fail(cuda::std::size_t n)
{
  T* ptr = static_cast<T*>(cuda::std::calloc(n, sizeof(T)));

  // NOTE(HIP/AMD): add a printf to avoid that the pointer is optimized out
  if(ptr != nullptr){
    printf("Error pointer %p is not a nullptr!\n", &ptr[0]);
  }

  // check that the memory was not allocated
  assert(ptr == nullptr);
}

struct BigStruct
{
  static constexpr cuda::std::size_t n = 32;

  int data[n];

  __host__ __device__ bool operator==(const BigStruct& other) const
  {
    for (cuda::std::size_t i{}; i < n; ++i)
    {
      if (data[i] != other.data[i])
      {
        return false;
      }
    }

    return true;
  }
};

struct alignas(cuda::std::max_align_t) AlignedStruct
{
  static constexpr cuda::std::size_t n = 32;

  char data[n];

  __host__ __device__ bool operator==(const AlignedStruct& other) const
  {
    for (cuda::std::size_t i{}; i < n; ++i)
    {
      if (data[i] != other.data[i])
      {
        return false;
      }
    }

    return true;
  }
};

__host__ __device__ void test()
{
  test_calloc_success<int>(10);
  test_calloc_success<char>(128);
  test_calloc_success<double>(8);
  test_calloc_success<BigStruct>(4);
  test_calloc_success<AlignedStruct>(16);

  test_calloc_fail<int>(cuda::std::numeric_limits<cuda::std::size_t>::max());
}

int main(int, char**)
{
  test();

  return 0;
}

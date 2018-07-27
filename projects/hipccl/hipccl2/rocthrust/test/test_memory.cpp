// MIT License
//
// Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
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

#include <iostream>
#include <type_traits>
#include <cstdlib>

// Google Test
#include <gtest/gtest.h>

// Thrust
#include <thrust/memory.h>

// HIP API
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
#include <hip/hip_runtime_api.h>

#define HIP_CHECK(condition) ASSERT_EQ(condition, hipSuccess)
#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

TEST(HipThrustMemory, VoidMalloc)
{
  const size_t n = 9001;
  thrust::device_system_tag dev_tag;

  using pointer = thrust::pointer<int, thrust::device_system_tag>;
  // Malloc on device
  auto void_ptr = thrust::malloc(dev_tag, sizeof(int) * n);
  pointer ptr = pointer(static_cast<int*>(void_ptr.get()));
  // Free
  thrust::free(dev_tag, ptr);
}

TEST(HipThrustMemory, TypeMalloc)
{
  const size_t n = 9001;
  thrust::device_system_tag dev_tag;

  // Malloc on device
  auto ptr = thrust::malloc<int>(dev_tag, sizeof(int) * n);
  // Free
  thrust::free(dev_tag, ptr);
}

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
TEST(HipThrustMemory, MallocUseMemory)
{
  const size_t n = 1024;
  thrust::device_system_tag dev_tag;

  // Malloc on device
  auto ptr = thrust::malloc<int>(dev_tag, sizeof(int) * n);

  // Try allocated memory with HIP function
  HIP_CHECK(hipMemset(ptr.get(), 0, n * sizeof(int)));

  // Free
  thrust::free(dev_tag, ptr);
}
#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

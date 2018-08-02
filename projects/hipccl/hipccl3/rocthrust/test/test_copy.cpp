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
#include <thrust/for_each.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>

// HIP API
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#define HIP_CHECK(condition) ASSERT_EQ(condition, hipSuccess)
#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
TEST(HipThrustCopy, HostToDevice)
{
  const size_t size = 256;
  thrust::device_system_tag dev_tag;
  thrust::host_system_tag host_tag;

  // Malloc on host
  auto h_ptr = thrust::malloc<int>(host_tag, sizeof(int) * size);
  // Malloc on device
  auto d_ptr = thrust::malloc<int>(dev_tag, sizeof(int) * size);

  for(size_t i = 0; i < size; i++)
  {
    *h_ptr = i;
  }

  // Compiles thanks to a temporary fix in
  // thrust/system/detail/generic/for_each.h
  thrust::copy(h_ptr, h_ptr + 256, d_ptr);

  // Free
  thrust::free(host_tag, h_ptr);
  thrust::free(dev_tag, d_ptr);
}

TEST(HipThrustCopy, DeviceToDevice)
{
  const size_t size = 256;
  thrust::device_system_tag dev_tag;

  // Malloc on device
  auto d_ptr1 = thrust::malloc<int>(dev_tag, sizeof(int) * size);
  auto d_ptr2 = thrust::malloc<int>(dev_tag, sizeof(int) * size);

  // Zero d_ptr1 memory
  HIP_CHECK(hipMemset(thrust::raw_pointer_cast(d_ptr1), 0, sizeof(int) * size));
  HIP_CHECK(hipMemset(thrust::raw_pointer_cast(d_ptr2), 0xdead, sizeof(int) * size));

  // Copy device->device
  thrust::copy(d_ptr1, d_ptr1 + 256, d_ptr2);

  std::vector<int> output(size);
  HIP_CHECK(
    hipMemcpy(
      output.data(), thrust::raw_pointer_cast(d_ptr2),
      size * sizeof(int),
      hipMemcpyDeviceToHost
    )
  );

  for(size_t i = 0; i < size; i++)
  {
    ASSERT_EQ(output[i], int(0)) << "where index = " << i;
  }

  // Free
  thrust::free(dev_tag, d_ptr1);
  thrust::free(dev_tag, d_ptr2);
}
#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

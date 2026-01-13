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
// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

// This test is currently failing as const_scalar_object cannot be used with get_device_address
// This is tracked in SWDEV-571304
// UNSUPPORTED: hipcc, hiprtc

#include <cuda/functional>
#include <cuda/std/cassert>

#include "test_macros.h"

__device__ int scalar_object             = 42;
__device__ const int const_scalar_object = 42;

__device__ int array_object[]             = {42, 1337, -1};
__device__ const int const_array_object[] = {42, 1337, -1};

#if !TEST_COMPILER(NVRTC) && !defined(TEST_COMPILER_HIPRTC)
template <class T>
void test_host(T& object)
{
  {
    T* host_address = cuda::std::addressof(object);

    cudaPointerAttributes attributes;
    cudaError_t status = cudaPointerGetAttributes(&attributes, host_address);
    assert(status == cudaSuccess);

    if (attributes.devicePointer)
    {
      assert(attributes.devicePointer == host_address);
    }
  }

  {
    T* device_address = cuda::get_device_address(object);

    cudaPointerAttributes attributes;
    cudaError_t status = cudaPointerGetAttributes(&attributes, device_address);
    assert(status == cudaSuccess);
    assert(attributes.devicePointer == device_address);
  }
}
#endif // !TEST_COMPILER(NVRTC)

template <class T>
__host__ __device__ void test(T& object)
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE, (assert(cuda::std::addressof(object) == cuda::get_device_address(object));), (test_host(object);))
}

int main(int argc, char** argv)
{
  test(scalar_object);
  test(const_scalar_object);
  test(array_object);
  test(const_array_object);

  return 0;
}

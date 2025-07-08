//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: nvrtc, hiprtc

#include <cuda/memory_resource>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/stream_ref>

enum class AccessibilityType
{
  Device,
  Host,
};

template <AccessibilityType Accessibilty>
struct resource
{
  void* allocate(size_t, size_t)
  {
    return nullptr;
  }
  void deallocate(void*, size_t, size_t) noexcept {}

  bool operator==(const resource&) const
  {
    return true;
  }
  bool operator!=(const resource& other) const
  {
    return false;
  }

  template <AccessibilityType Accessibilty2                                         = Accessibilty,
            cuda::std::enable_if_t<Accessibilty2 == AccessibilityType::Device, int> = 0>
  friend void get_property(const resource&, cuda::mr::device_accessible) noexcept
  {}
};
static_assert(cuda::mr::resource<resource<AccessibilityType::Host>>, "");
static_assert(!cuda::mr::resource_with<resource<AccessibilityType::Host>, cuda::mr::device_accessible>, "");
static_assert(cuda::mr::resource<resource<AccessibilityType::Device>>, "");
static_assert(cuda::mr::resource_with<resource<AccessibilityType::Device>, cuda::mr::device_accessible>, "");

template <AccessibilityType Accessibilty>
struct async_resource : public resource<Accessibilty>
{
  void* allocate_async(size_t, size_t, cuda::stream_ref)
  {
    return nullptr;
  }
  void deallocate_async(void*, size_t, size_t, cuda::stream_ref) {}
};
static_assert(cuda::mr::async_resource<async_resource<AccessibilityType::Host>>, "");
static_assert(!cuda::mr::async_resource_with<async_resource<AccessibilityType::Host>, cuda::mr::device_accessible>, "");
static_assert(cuda::mr::async_resource<async_resource<AccessibilityType::Device>>, "");
static_assert(cuda::mr::async_resource_with<async_resource<AccessibilityType::Device>, cuda::mr::device_accessible>,
              "");

// test for cccl#2214: https://github.com/NVIDIA/cccl/issues/2214
struct derived_resource : cuda::mr::device_memory_resource
{
  using cuda::mr::device_memory_resource::device_memory_resource;
};
static_assert(cuda::mr::resource<derived_resource>, "");

// Ensure that we can only

void test()
{
  cuda::mr::device_memory_resource first{};
  { // comparison against a plain device_memory_resource
    cuda::mr::device_memory_resource second{};
    assert(first == second);
    assert(!(first != second));
  }

  { // comparison against a device_memory_resource wrapped inside a resource_ref<device_accessible>
    cuda::mr::device_memory_resource second{};
    cuda::mr::resource_ref<cuda::mr::device_accessible> second_ref{second};
    assert(first == second_ref);
    assert(!(first != second_ref));
    assert(second_ref == first);
    assert(!(second_ref != first));
  }

  { // comparison against a device_memory_resource wrapped inside a resource_ref<>
    cuda::mr::device_memory_resource second{};
    cuda::mr::resource_ref<> second_ref{second};
    assert(first == second_ref);
    assert(!(first != second_ref));
    assert(second_ref == first);
    assert(!(second_ref != first));
  }

  { // comparison against a different resource
    resource<AccessibilityType::Host> host_resource{};
    resource<AccessibilityType::Device> device_resource{};
    assert(!(first == host_resource));
    assert(first != host_resource);
    assert(!(first == device_resource));
    assert(first != device_resource);

    assert(!(host_resource == first));
    assert(host_resource != first);
    assert(!(device_resource == first));
    assert(device_resource != first);
  }

  { // comparison against a different resource through resource_ref
    async_resource<AccessibilityType::Host> host_async_resource{};
    async_resource<AccessibilityType::Device> device_async_resource{};
    cuda::mr::resource_ref<> host_ref{host_async_resource};
    cuda::mr::resource_ref<> device_ref{device_async_resource};
    assert(!(first == host_ref));
    assert(first != host_ref);
    assert(!(first == device_async_resource));
    assert(first != device_async_resource);

    assert(!(host_ref == first));
    assert(host_ref != first);
    assert(!(device_async_resource == first));
    assert(device_async_resource != first);
  }
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, test();)
  return 0;
}

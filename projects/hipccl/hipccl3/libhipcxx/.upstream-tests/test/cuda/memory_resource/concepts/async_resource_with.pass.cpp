//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
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

// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: nvrtc, hiprtc

// cuda::mr::async_resource_with

#include <cuda/memory_resource>
#include <cuda/std/cstdint>

struct prop_with_value
{};
struct prop
{};

struct valid_resource_with_property
{
  void* allocate(std::size_t, std::size_t)
  {
    return nullptr;
  }
  void deallocate(void*, std::size_t, std::size_t) noexcept {}
  void* allocate_async(std::size_t, std::size_t, cuda::stream_ref)
  {
    return nullptr;
  }
  void deallocate_async(void*, std::size_t, std::size_t, cuda::stream_ref) {}
  bool operator==(const valid_resource_with_property&) const
  {
    return true;
  }
  bool operator!=(const valid_resource_with_property&) const
  {
    return false;
  }
  friend void get_property(const valid_resource_with_property&, prop_with_value) {}
};
static_assert(cuda::mr::async_resource_with<valid_resource_with_property, prop_with_value>, "");

struct valid_resource_without_property
{
  void* allocate(std::size_t, std::size_t)
  {
    return nullptr;
  }
  void deallocate(void*, std::size_t, std::size_t) noexcept {}
  void* allocate_async(std::size_t, std::size_t, cuda::stream_ref)
  {
    return nullptr;
  }
  void deallocate_async(void*, std::size_t, std::size_t, cuda::stream_ref) {}
  bool operator==(const valid_resource_without_property&) const
  {
    return true;
  }
  bool operator!=(const valid_resource_without_property&) const
  {
    return false;
  }
};
static_assert(!cuda::mr::async_resource_with<valid_resource_without_property, prop_with_value>, "");

struct invalid_resource_with_property
{
  friend void get_property(const invalid_resource_with_property&, prop_with_value) {}
};
static_assert(!cuda::mr::async_resource_with<invalid_resource_with_property, prop_with_value>, "");

struct resource_with_many_properties
{
  void* allocate(std::size_t, std::size_t)
  {
    return nullptr;
  }
  void deallocate(void*, std::size_t, std::size_t) noexcept {}
  void* allocate_async(std::size_t, std::size_t, cuda::stream_ref)
  {
    return nullptr;
  }
  void deallocate_async(void*, std::size_t, std::size_t, cuda::stream_ref) {}
  bool operator==(const resource_with_many_properties&) const
  {
    return true;
  }
  bool operator!=(const resource_with_many_properties&) const
  {
    return false;
  }
  friend void get_property(const resource_with_many_properties&, prop_with_value) {}
  friend void get_property(const resource_with_many_properties&, prop) {}
};
static_assert(cuda::mr::async_resource_with<resource_with_many_properties, prop_with_value, prop>, "");
static_assert(!cuda::mr::async_resource_with<resource_with_many_properties, prop_with_value, int, prop>, "");

struct derived_with_property : public valid_resource_without_property
{
  friend void get_property(const derived_with_property&, prop_with_value) {}
};
static_assert(cuda::mr::async_resource_with<derived_with_property, prop_with_value>, "");

int main(int, char**)
{
  return 0;
}

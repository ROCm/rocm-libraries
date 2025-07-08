//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
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

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: nvrtc, hiprtc

// cuda::mr::resource_ref properties

#include <cuda/memory_resource>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

template <class T>
struct property_with_value
{
  using value_type = T;
};

template <class T>
struct property_without_value
{};

struct Fake_alloc_base
{
  void* object                                       = nullptr;
  const cuda::mr::_Async_alloc_vtable* static_vtable = nullptr;
};

template <class... Properties>
struct resource
{
  void* allocate(std::size_t, std::size_t)
  {
    return &_val;
  }

  void deallocate(void* ptr, std::size_t, std::size_t) noexcept
  {
    // ensure that we did get the right inputs forwarded
    _val = *static_cast<int*>(ptr);
  }

  void* allocate_async(std::size_t, std::size_t, cuda::stream_ref)
  {
    return &_val;
  }

  void deallocate_async(void* ptr, std::size_t, std::size_t, cuda::stream_ref)
  {
    // ensure that we did get the right inputs forwarded
    _val = *static_cast<int*>(ptr);
  }

  bool operator==(const resource& other) const
  {
    return _val == other._val;
  }
  bool operator!=(const resource& other) const
  {
    return _val != other._val;
  }

  int _val = 0;

  _LIBCUDACXX_TEMPLATE(class Property)
  _LIBCUDACXX_REQUIRES((!cuda::property_with_value<Property>) && _CUDA_VSTD::_One_of<Property, Properties...>) //
  friend void get_property(const resource&, Property) noexcept {}

  _LIBCUDACXX_TEMPLATE(class Property)
  _LIBCUDACXX_REQUIRES(cuda::property_with_value<Property>&& _CUDA_VSTD::_One_of<Property, Properties...>) //
  friend typename Property::value_type get_property(const resource& res, Property) noexcept
  {
    return static_cast<typename Property::value_type>(res._val);
  }
};

template <class PropA, class PropB>
void test_conversion_from_async_resource_ref()
{
  resource<PropA, PropB> input{42};
  cuda::mr::async_resource_ref<PropA, PropB> ref_input{input};

  { // lvalue
    cuda::mr::async_resource_ref<PropB> ref{ref_input};

    // Ensure that we properly "punch through" the resource ref
    const auto fake_orig = reinterpret_cast<Fake_alloc_base*>(&ref_input);
    const auto fake_conv = reinterpret_cast<Fake_alloc_base*>(&ref);
    assert(fake_orig->object == fake_conv->object);
    assert(fake_orig->static_vtable == fake_conv->static_vtable);

    // Ensure that we properly pass on the allocate function
    assert(input.allocate_async(0, 0, {}) == ref.allocate_async(0, 0, {}));

    // Ensure we are deallocating properly
    int expected_after_deallocate = 1337;
    ref.deallocate_async(static_cast<void*>(&expected_after_deallocate), 0, 0, {});
    assert(input._val == expected_after_deallocate);
  }

  { // prvalue
    cuda::mr::async_resource_ref<PropB> ref{cuda::mr::async_resource_ref<PropA, PropB>{input}};

    // Ensure that we properly "punch through" the resource ref
    const auto fake_orig = reinterpret_cast<Fake_alloc_base*>(&ref_input);
    const auto fake_conv = reinterpret_cast<Fake_alloc_base*>(&ref);
    assert(fake_orig->object == fake_conv->object);
    assert(fake_orig->static_vtable == fake_conv->static_vtable);

    // Ensure that we properly pass on the allocate function
    assert(input.allocate_async(0, 0, {}) == ref.allocate_async(0, 0, {}));

    // Ensure we are deallocating properly
    int expected_after_deallocate = 1337;
    ref.deallocate_async(static_cast<void*>(&expected_after_deallocate), 0, 0, {});
    assert(input._val == expected_after_deallocate);
  }
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST,
               (test_conversion_from_async_resource_ref<property_with_value<short>, property_with_value<int>>();
                test_conversion_from_async_resource_ref<property_with_value<short>, property_without_value<int>>();
                test_conversion_from_async_resource_ref<property_without_value<short>, property_without_value<int>>();))

  return 0;
}

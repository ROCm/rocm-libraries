//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

// cuda::mr::async_resource_ref construction

#include <cuda/memory_resource>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include "types.h"

namespace constructible
{
using ref = cuda::mr::async_resource_ref<cuda::mr::host_accessible,
                                         property_with_value<int>,
                                         property_with_value<double>,
                                         property_without_value<std::size_t>>;

using matching_properties =
  async_resource<cuda::mr::host_accessible,
                 property_with_value<double>,
                 property_without_value<std::size_t>,
                 property_with_value<int>>;

using missing_stateful_property =
  async_resource<cuda::mr::host_accessible, property_with_value<int>, property_without_value<std::size_t>>;
using missing_stateless_property =
  async_resource<cuda::mr::host_accessible, property_with_value<int>, property_with_value<double>>;

using cuda::std::is_constructible;
static_assert(is_constructible<ref, matching_properties&>::value, "");
static_assert(!is_constructible<ref, missing_stateful_property&>::value, "");
static_assert(!is_constructible<ref, missing_stateless_property&>::value, "");

static_assert(is_constructible<ref, matching_properties*>::value, "");
static_assert(!is_constructible<ref, missing_stateful_property*>::value, "");
static_assert(!is_constructible<ref, missing_stateless_property*>::value, "");

static_assert(is_constructible<ref, ref&>::value, "");

// Ensure we require a mutable valid reference and do not bind against rvalues
static_assert(!is_constructible<ref, matching_properties>::value, "");
static_assert(!is_constructible<ref, const matching_properties&>::value, "");
static_assert(!is_constructible<ref, const matching_properties*>::value, "");

static_assert(cuda::std::is_copy_constructible<ref>::value, "");
static_assert(cuda::std::is_move_constructible<ref>::value, "");
} // namespace constructible

namespace assignable
{
using ref = cuda::mr::async_resource_ref<cuda::mr::host_accessible,
                                         property_with_value<int>,
                                         property_with_value<double>,
                                         property_without_value<std::size_t>>;

using res = async_resource<cuda::mr::host_accessible,
                           property_with_value<int>,
                           property_with_value<double>,
                           property_without_value<std::size_t>>;

using other_res =
  async_resource<cuda::mr::host_accessible,
                 property_without_value<int>,
                 property_with_value<int>,
                 property_with_value<double>,
                 property_without_value<std::size_t>>;

using cuda::std::is_assignable;
static_assert(cuda::std::is_assignable<ref, res&>::value, "");
static_assert(cuda::std::is_assignable<ref, other_res&>::value, "");

static_assert(cuda::std::is_copy_assignable<ref>::value, "");
static_assert(cuda::std::is_move_assignable<ref>::value, "");
} // namespace assignable

int main(int, char**)
{
  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: nvrtc
// UNSUPPORTED: hiprtc

#include <cuda/memory_resource>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

// Verify that the properties exist
static_assert(cuda::std::is_empty<cuda::mr::host_accessible>::value, "");
static_assert(cuda::std::is_empty<cuda::mr::device_accessible>::value, "");

// Verify that host accessible is the default if nothing is specified
static_assert(!cuda::mr::__is_host_accessible<>, "");
static_assert(cuda::mr::__is_host_accessible<cuda::mr::host_accessible>, "");
static_assert(!cuda::mr::__is_host_accessible<cuda::mr::device_accessible>, "");
static_assert(cuda::mr::__is_host_accessible<cuda::mr::host_accessible, cuda::mr::device_accessible>, "");

// Verify that device accessible needs to be explicitly specified
static_assert(!cuda::mr::__is_device_accessible<>, "");
static_assert(!cuda::mr::__is_device_accessible<cuda::mr::host_accessible>, "");
static_assert(cuda::mr::__is_device_accessible<cuda::mr::device_accessible>, "");
static_assert(cuda::mr::__is_device_accessible<cuda::mr::host_accessible, cuda::mr::device_accessible>, "");

// Verify that host device accessible needs to be explicitly specified
static_assert(!cuda::mr::__is_host_device_accessible<>, "");
static_assert(!cuda::mr::__is_host_device_accessible<cuda::mr::host_accessible>, "");
static_assert(!cuda::mr::__is_host_device_accessible<cuda::mr::device_accessible>, "");
static_assert(cuda::mr::__is_host_device_accessible<cuda::mr::host_accessible, cuda::mr::device_accessible>, "");

int main(int, char**)
{
  return 0;
}

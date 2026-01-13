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

// cuda::mr::resource_ref equality

#include <cuda/memory_resource>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

#include "types.h"

using ref = cuda::mr::resource_ref<cuda::mr::host_accessible,
                                   property_with_value<int>,
                                   property_with_value<double>,
                                   property_without_value<std::size_t>>;

using pertubed_properties =
  cuda::mr::resource_ref<cuda::mr::host_accessible,
                         property_with_value<double>,
                         property_with_value<int>,
                         property_without_value<std::size_t>>;

using res = resource<cuda::mr::host_accessible,
                     property_with_value<int>,
                     property_with_value<double>,
                     property_without_value<std::size_t>>;
using other_res =
  resource<cuda::mr::host_accessible,
           property_with_value<double>,
           property_with_value<int>,
           property_without_value<std::size_t>>;

void test_equality()
{
  res input{42};
  res with_equal_value{42};
  res with_different_value{1337};

  assert(input == with_equal_value);
  assert(input != with_different_value);

  assert(ref{input} == ref{with_equal_value});
  assert(ref{input} != ref{with_different_value});

  // Should ignore pertubed properties
  assert(ref{input} == pertubed_properties{with_equal_value});
  assert(ref{input} != pertubed_properties{with_different_value});

  // Should reject different resources
  other_res other_with_matching_value{42};
  other_res other_with_different_value{1337};
  assert(ref{input} != ref{other_with_matching_value});
  assert(ref{input} != ref{other_with_different_value});

  assert(ref{input} != pertubed_properties{other_with_matching_value});
  assert(ref{input} != pertubed_properties{other_with_matching_value});
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test_equality();))

  return 0;
}

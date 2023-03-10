//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

// UNSUPPORTED: hipcc
// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

// This test ensures that cuda::std::memory_order has the same size under all
// standard versions to make sure we're not breaking the ABI. This is
// relevant because cuda::std::memory_order is a scoped enumeration in C++20,
// but an unscoped enumeration pre-C++20.
//
// See PR40977 for details.

#include <cuda/std/atomic>
#include <cuda/std/type_traits>

#include "test_macros.h"

enum cpp17_memory_order
{
  cpp17_memory_order_relaxed,
  cpp17_memory_order_consume,
  cpp17_memory_order_acquire,
  cpp17_memory_order_release,
  cpp17_memory_order_acq_rel,
  cpp17_memory_order_seq_cst
};

static_assert((cuda::std::is_same<cuda::std::underlying_type<cpp17_memory_order>::type,
                                  cuda::std::underlying_type<cuda::std::memory_order>::type>::value),
              "cuda::std::memory_order should have the same underlying type as a corresponding "
              "unscoped enumeration would. Otherwise, our ABI changes from C++17 to C++20.");

int main(int, char**)
{
  return 0;
}

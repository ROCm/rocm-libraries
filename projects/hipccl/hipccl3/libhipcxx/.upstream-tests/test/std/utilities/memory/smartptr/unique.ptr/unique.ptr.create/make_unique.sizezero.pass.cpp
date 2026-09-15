//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
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

// This code triggers https://gcc.gnu.org/bugzilla/show_bug.cgi?id=104568
// UNSUPPORTED: msvc
// UNSUPPORTED: nvrtc, hiprtc
// UNSUPPORTED: nvhpc

// Test the fix for https://llvm.org/PR54100

#include <cuda/std/__memory_>
#include <cuda/std/cassert>

#include "test_macros.h"

struct A
{
  int m[0];
};
static_assert(sizeof(A) == 0, ""); // an extension supported by GCC and Clang

int main(int, char**)
{
  {
    cuda::std::unique_ptr<A> p = cuda::std::unique_ptr<A>(new A);
    assert(p != nullptr);
  }
  {
    cuda::std::unique_ptr<A[]> p = cuda::std::unique_ptr<A[]>(new A[1]);
    assert(p != nullptr);
  }
  {
    cuda::std::unique_ptr<A> p = cuda::std::make_unique<A>();
    assert(p != nullptr);
  }
  {
    cuda::std::unique_ptr<A[]> p = cuda::std::make_unique<A[]>(1);
    assert(p != nullptr);
  }

  return 0;
}

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

// test forward

#include <cuda/std/utility>

#include "test_macros.h"

struct A
{};

__host__ __device__ A source()
{
  return A();
}
__host__ __device__ const A csource()
{
  return A();
}

int main(int, char**)
{
  {
    (void) cuda::std::forward<A&>(source()); // expected-note {{requested here}}
    // expected-error-re@__utility/forward.h:* {{{{(static_assert|static assertion)}} failed{{.*}} {{"?}}cannot forward
    // an rvalue as an lvalue{{"?}}}}
#if TEST_COMPILER(CLANG, >, 14) && !defined(TEST_COMPILER_HIPCC)  // NOTE(HIP): hipcc does not emit this error message
    // expected-error {{ignoring return value of function declared with const attribute}}
#endif // TEST_COMPILER(CLANG, >, 14)
  }
  {
    const A ca = A();
    cuda::std::forward<A&>(ca); // expected-error {{no matching function for call to 'forward'}}
  }
  {
    cuda::std::forward<A&>(csource()); // expected-error {{no matching function for call to 'forward'}}
  }
  {
    const A ca = A();
    cuda::std::forward<A>(ca); // expected-error {{no matching function for call to 'forward'}}
  }
  {
    cuda::std::forward<A>(csource()); // expected-error {{no matching function for call to 'forward'}}
  }
  {
    A a;
    cuda::std::forward(a); // expected-error {{no matching function for call to 'forward'}}
  }

  return 0;
}

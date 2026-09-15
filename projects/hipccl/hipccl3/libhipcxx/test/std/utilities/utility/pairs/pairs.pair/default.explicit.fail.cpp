//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

// Before GCC 6, aggregate initialization kicks in.
// See https://stackoverflow.com/q/41799015/627587.
// UNSUPPORTED: gcc-5

// <utility>

// template <class T1, class T2> struct pair

// explicit(see-below) constexpr pair();

// This test checks the conditional explicitness of cuda::std::pair's default
// constructor as introduced by the resolution of LWG 2510.

#include <cuda/std/utility>

struct ImplicitlyDefaultConstructible
{
  ImplicitlyDefaultConstructible() = default;
};

struct ExplicitlyDefaultConstructible
{
  explicit ExplicitlyDefaultConstructible() = default;
};

cuda::std::pair<ImplicitlyDefaultConstructible, ExplicitlyDefaultConstructible> test1()
{
  return {};
} // expected-error-re 1 {{chosen constructor is explicit in copy-initialization}}
cuda::std::pair<ExplicitlyDefaultConstructible, ImplicitlyDefaultConstructible> test2()
{
  return {};
} // expected-error-re 1 {{chosen constructor is explicit in copy-initialization}}
cuda::std::pair<ExplicitlyDefaultConstructible, ExplicitlyDefaultConstructible> test3()
{
  return {};
} // expected-error-re 1 {{chosen constructor is explicit in copy-initialization}}
cuda::std::pair<ImplicitlyDefaultConstructible, ImplicitlyDefaultConstructible> test4()
{
  return {};
}

int main(int, char**)
{
  return 0;
}

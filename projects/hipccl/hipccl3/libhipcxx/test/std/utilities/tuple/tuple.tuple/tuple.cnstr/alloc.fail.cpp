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

// FIXME(HIP): fix following compiler error: /usr/lib/gcc/x86_64-linux-gnu/9/../../../../include/c++/9/tuple:699:2: note: candidate constructor template not viable: no known conversion from 'const cuda::std::allocator_arg_t' to 'std::allocator_arg_t' for 1st argument tuple(allocator_arg_t __tag, const _Alloc& __a)

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class Alloc>
//   explicit(see-below) tuple(allocator_arg_t, const Alloc& a);

// Make sure we get the explicit-ness of the constructor right.
// This is LWG 3158.

#include <cuda/std/tuple>

struct ExplicitDefault
{
  __host__ __device__ explicit ExplicitDefault() {}
};

__host__ __device__ std::tuple<ExplicitDefault> explicit_default_test()
{
  return {cuda::std::allocator_arg, cuda::std::allocator<int>()}; // expected-error {{chosen constructor is explicit in
                                                                  // copy-initialization}}
}

int main(int, char**)
{
  return 0;
}

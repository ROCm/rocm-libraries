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

// <cuda/std/tuple>

// template <class... Types> class tuple;

// explicit(see-below) constexpr tuple();

#include <cuda/std/tuple>

struct Implicit
{
  Implicit() = default;
};

struct Explicit
{
  explicit Explicit() = default;
};

__host__ __device__ cuda::std::tuple<> test1()
{
  return {};
}

__host__ __device__ cuda::std::tuple<Implicit> test2()
{
  return {};
}
__host__ __device__ cuda::std::tuple<Explicit> test3()
{
  return {};
} // expected-error-re 1 {{chosen constructor is explicit in copy-initialization}}

__host__ __device__ cuda::std::tuple<Implicit, Implicit> test4()
{
  return {};
}
__host__ __device__ cuda::std::tuple<Explicit, Implicit> test5()
{
  return {};
} // expected-error-re 1 {{chosen constructor is explicit in copy-initialization}}
__host__ __device__ cuda::std::tuple<Implicit, Explicit> test6()
{
  return {};
} // expected-error-re 1 {{chosen constructor is explicit in copy-initialization}}
__host__ __device__ cuda::std::tuple<Explicit, Explicit> test7()
{
  return {};
} // expected-error-re 1 {{chosen constructor is explicit in copy-initialization}}

__host__ __device__ cuda::std::tuple<Implicit, Implicit, Implicit> test8()
{
  return {};
}
__host__ __device__ cuda::std::tuple<Implicit, Implicit, Explicit> test9()
{
  return {};
} // expected-error-re 1 {{chosen constructor is explicit in copy-initialization}}
__host__ __device__ cuda::std::tuple<Implicit, Explicit, Implicit> test10()
{
  return {};
} // expected-error-re 1 {{chosen constructor is explicit in copy-initialization}}
__host__ __device__ cuda::std::tuple<Implicit, Explicit, Explicit> test11()
{
  return {};
} // expected-error-re 1 {{chosen constructor is explicit in copy-initialization}}
__host__ __device__ cuda::std::tuple<Explicit, Implicit, Implicit> test12()
{
  return {};
} // expected-error-re 1 {{chosen constructor is explicit in copy-initialization}}
__host__ __device__ cuda::std::tuple<Explicit, Implicit, Explicit> test13()
{
  return {};
} // expected-error-re 1 {{chosen constructor is explicit in copy-initialization}}
__host__ __device__ cuda::std::tuple<Explicit, Explicit, Implicit> test14()
{
  return {};
} // expected-error-re 1 {{chosen constructor is explicit in copy-initialization}}
__host__ __device__ cuda::std::tuple<Explicit, Explicit, Explicit> test15()
{
  return {};
} // expected-error-re 1 {{chosen constructor is explicit in copy-initialization}}

int main(int, char**)
{
  return 0;
}

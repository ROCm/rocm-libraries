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
// UNSUPPORTED: c++98, c++03
// UNSUPPORTED: msvc

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class Alloc> tuple(allocator_arg_t, Alloc const&)

// Libc++ has to deduce the 'allocator_arg_t' parameter for this constructor
// as 'AllocArgT'. Previously libc++ has tried to support tags derived from
// 'allocator_arg_t' by using 'is_base_of<AllocArgT, allocator_arg_t>'.
// However this breaks whenever a 2-tuple contains a reference to an incomplete
// type as its first parameter. See PR27684.

#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "test_macros.h"

struct IncompleteType;
TEST_ACCESSIBLE extern IncompleteType inc1;
TEST_ACCESSIBLE extern IncompleteType inc2;
TEST_ACCESSIBLE IncompleteType const& cinc1 = inc1;
TEST_ACCESSIBLE IncompleteType const& cinc2 = inc2;

int main(int, char**)
{
  using IT = IncompleteType;
  { // try calling tuple(Tp const&...)
    using Tup = cuda::std::tuple<const IT&, const IT&>;
    Tup t(cinc1, cinc2);
    assert(&cuda::std::get<0>(t) == &inc1);
    assert(&cuda::std::get<1>(t) == &inc2);
  }
  { // try calling tuple(Up&&...)
    using Tup = cuda::std::tuple<const IT&, const IT&>;
    Tup t(inc1, inc2);
    assert(&cuda::std::get<0>(t) == &inc1);
    assert(&cuda::std::get<1>(t) == &inc2);
  }

  return 0;
}

struct IncompleteType
{};
TEST_ACCESSIBLE IncompleteType inc1;
TEST_ACCESSIBLE IncompleteType inc2;

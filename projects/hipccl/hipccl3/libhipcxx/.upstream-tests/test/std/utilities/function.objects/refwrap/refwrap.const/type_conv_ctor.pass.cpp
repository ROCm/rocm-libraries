//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
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

// <functional>
//
// reference_wrapper
//
// template <class U>
//   reference_wrapper(U&&) noexcept(see below);

// #include <cuda/std/functional>
#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "test_macros.h"

struct convertible_to_int_ref
{
  int val = 0;
  __host__ __device__ operator int&()
  {
    return val;
  }
  __host__ __device__ operator int const&() const
  {
    return val;
  }
};

template <bool IsNothrow>
struct nothrow_convertible
{
  int val = 0;
  __host__ __device__ operator int&() noexcept(IsNothrow)
  {
    return val;
  }
};

struct convertible_from_int
{
  __host__ __device__ convertible_from_int(int) {}
};

__host__ __device__ void meow(cuda::std::reference_wrapper<int>) {}
__host__ __device__ void meow(convertible_from_int) {}

int main(int, char**)
{
  {
    convertible_to_int_ref t;
    cuda::std::reference_wrapper<convertible_to_int_ref> r(t);
    assert(&r.get() == &t);
  }
  {
    const convertible_to_int_ref t{};
    cuda::std::reference_wrapper<const convertible_to_int_ref> r(t);
    assert(&r.get() == &t);
  }
  {
    using Ref = cuda::std::reference_wrapper<int>;
    static_assert(noexcept(Ref(nothrow_convertible<true>())));
#if !TEST_COMPILER(NVHPC)
    static_assert(!noexcept(Ref(nothrow_convertible<false>())));
#endif // !TEST_COMPILER(NVHPC)
  }
  {
    meow(0);
  }
#if !TEST_COMPILER(MSVC) && !TEST_COMPILER(NVRTC) && !defined(TEST_COMPILER_HIPRTC)
  {
    extern cuda::std::reference_wrapper<int> purr();
    static_assert(cuda::std::is_same_v<decltype(true ? purr() : 0), int>);
  }
#endif // !TEST_COMPILER(MSVC)
#if !TEST_COMPILER(GCC, <, 8) // gcc-7 is broken wrt ctad
  {
    int i = 0;
    cuda::std::reference_wrapper ri(i);
    static_assert((cuda::std::is_same<decltype(ri), cuda::std::reference_wrapper<int>>::value), "");
    const int j = 0;
    cuda::std::reference_wrapper rj(j);
    static_assert((cuda::std::is_same<decltype(rj), cuda::std::reference_wrapper<const int>>::value), "");
  }
#endif // !TEST_COMPILER(GCC, <, 8)

  return 0;
}

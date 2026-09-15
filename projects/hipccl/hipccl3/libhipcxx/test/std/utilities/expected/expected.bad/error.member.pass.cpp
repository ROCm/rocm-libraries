//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
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

// UNSUPPORTED: libcpp-no-exceptions
// UNSUPPORTED: nvrtc, hiprtc

// E& error() & noexcept;
// const E& error() const & noexcept;
// E&& error() && noexcept;
// const E&& error() const && noexcept;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/expected>
#include <cuda/std/utility>

template <class T, class = void>
constexpr bool ErrorNoexcept = false;

template <class T>
constexpr bool ErrorNoexcept<T, cuda::std::void_t<decltype(cuda::std::declval<T&&>().error())>> =
  noexcept(cuda::std::declval<T&&>().error());

static_assert(!ErrorNoexcept<int>, "");
static_assert(ErrorNoexcept<cuda::std::bad_expected_access<int>&>, "");
static_assert(ErrorNoexcept<cuda::std::bad_expected_access<int> const&>, "");
static_assert(ErrorNoexcept<cuda::std::bad_expected_access<int>&&>, "");
static_assert(ErrorNoexcept<cuda::std::bad_expected_access<int> const&&>, "");

void test()
{
  // &
  {
    cuda::std::bad_expected_access<int> e(5);
    decltype(auto) i = e.error();
    static_assert(cuda::std::same_as<decltype(i), int&>, "");
    assert(i == 5);
  }

  // const &
  {
    const cuda::std::bad_expected_access<int> e(5);
    decltype(auto) i = e.error();
    static_assert(cuda::std::same_as<decltype(i), const int&>, "");
    assert(i == 5);
  }

  // &&
  {
    cuda::std::bad_expected_access<int> e(5);
    decltype(auto) i = cuda::std::move(e).error();
    static_assert(cuda::std::same_as<decltype(i), int&&>, "");
    assert(i == 5);
  }

  // const &&
  {
    const cuda::std::bad_expected_access<int> e(5);
    decltype(auto) i = cuda::std::move(e).error();
    static_assert(cuda::std::same_as<decltype(i), const int&&>, "");
    assert(i == 5);
  }
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}

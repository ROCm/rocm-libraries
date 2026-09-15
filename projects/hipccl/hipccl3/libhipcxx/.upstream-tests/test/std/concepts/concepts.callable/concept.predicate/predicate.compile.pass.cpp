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

// template<class F, class... Args>
// concept predicate;

#include <cuda/std/concepts>
#include <cuda/std/type_traits>

#include "test_macros.h"

using cuda::std::predicate;

static_assert(predicate<bool()>, "");
static_assert(predicate<bool (*)()>, "");
static_assert(predicate<bool (&)()>, "");

static_assert(!predicate<void()>, "");
static_assert(!predicate<void (*)()>, "");
static_assert(!predicate<void (&)()>, "");

struct S
{};

static_assert(!predicate<S(int), int>, "");
static_assert(!predicate<S(double), double>, "");
static_assert(predicate<int S::*, S*>, "");
static_assert(predicate<int (S::*)(), S*>, "");
static_assert(predicate<int (S::*)(), S&>, "");
static_assert(!predicate<void (S::*)(), S*>, "");
static_assert(!predicate<void (S::*)(), S&>, "");

static_assert(!predicate<bool(S)>, "");
static_assert(!predicate<bool(S)>, "");
#if !TEST_COMPILER(MSVC) || TEST_STD_VER > 2017 // unspecified MSVC bug
static_assert(!predicate<bool(S&), S>, "");
#endif // !TEST_COMPILER(MSVC) || TEST_STD_VER > 2017
static_assert(!predicate<bool(S&), S const&>, "");
static_assert(predicate<bool(S&), S&>, "");

struct Predicate
{
  __host__ __device__ bool operator()(int, double, char);
};
static_assert(predicate<Predicate, int, double, char>, "");
static_assert(predicate<Predicate&, int, double, char>, "");
static_assert(!predicate<const Predicate, int, double, char>, "");
static_assert(!predicate<const Predicate&, int, double, char>, "");

#if !defined(TEST_COMPILER_NVRTC) && !defined(TEST_COMPILER_HIPRTC)
template <class Fun>
__host__ __device__ constexpr bool check_lambda(Fun)
{
  return predicate<Fun>;
}

static_assert(check_lambda([] {
                return cuda::std::true_type();
              }),
              "");
static_assert(check_lambda([]() -> int* {
                return nullptr;
              }),
              "");

struct boolean
{
  __host__ __device__ operator bool() const noexcept;
};
static_assert(check_lambda([] {
                return boolean();
              }),
              "");

struct explicit_bool
{
  __host__ __device__ explicit operator bool() const noexcept;
};
static_assert(!check_lambda([] {
  return explicit_bool();
}),
              "");
#endif // !TEST_COMPILER(NVRTC)

int main(int, char**)
{
  return 0;
}

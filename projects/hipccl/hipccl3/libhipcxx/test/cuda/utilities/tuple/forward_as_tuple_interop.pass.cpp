//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
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

// UNSUPPORTED: nvrtc, hiprtc
#include <cuda/std/cassert>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include <tuple>

#include <nv/target>

constexpr bool test()
{
  // Ensure we can use std:: types inside cuda::std::make_tuple
  {
    using ret = cuda::std::tuple<cuda::std::integral_constant<int, 42>, std::integral_constant<int, 1337>>;
    auto t    = cuda::std::make_tuple(cuda::std::integral_constant<int, 42>(), std::integral_constant<int, 1337>());
    static_assert(cuda::std::is_same<decltype(t), ret>::value, "");
    assert(cuda::std::get<0>(t) == 42);
    assert(cuda::std::get<1>(t) == 1337);
  }

  // Ensure we can use std:: types inside cuda::std::tuple_cat
  {
    using ret = cuda::std::tuple<cuda::std::integral_constant<int, 42>, std::integral_constant<int, 1337>>;
    auto t    = cuda::std::tuple_cat(cuda::std::make_tuple(cuda::std::integral_constant<int, 42>()),
                                  cuda::std::make_tuple(std::integral_constant<int, 1337>()));
    static_assert(cuda::std::is_same<decltype(t), ret>::value, "");
    assert(cuda::std::get<0>(t) == 42);
    assert(cuda::std::get<1>(t) == 1337);
  }

  return true;
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test(); static_assert(test(), "");));

  return 0;
}

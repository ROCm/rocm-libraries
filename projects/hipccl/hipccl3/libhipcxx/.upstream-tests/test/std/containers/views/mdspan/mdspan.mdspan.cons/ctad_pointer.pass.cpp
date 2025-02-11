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

// UNSUPPORTED: msvc && c++14, msvc && c++17

// No CTAD in C++14 or earlier

#include <cuda/std/cassert>
#include <cuda/std/mdspan>

#define CHECK_MDSPAN(m, d)                                                      \
  static_assert(cuda::std::is_same<decltype(m)::element_type, int>::value, ""); \
  static_assert(m.is_exhaustive(), "");                                         \
  assert(m.data_handle() == d.data());                                          \
  assert(m.rank() == 0);                                                        \
  assert(m.rank_dynamic() == 0)

int main(int, char**)
{
#ifdef __MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION
  // TEST(TestMdspanCTAD, ctad_pointer)
  {
    cuda::std::array<int, 5> d = {1, 2, 3, 4, 5};
    int* ptr                   = d.data();
    cuda::std::mdspan m(ptr);

    CHECK_MDSPAN(m, d);
  }

  // TEST(TestMdspanCTAD, ctad_pointer_tmp)
  {
    cuda::std::array<int, 5> d = {1, 2, 3, 4, 5};
    cuda::std::mdspan m(d.data());

    CHECK_MDSPAN(m, d);
  }

  // TEST(TestMdspanCTAD, ctad_pointer_move)
  {
    cuda::std::array<int, 5> d = {1, 2, 3, 4, 5};
    int* ptr                   = d.data();
    // NOTE(HIPRTC): Internal issue #83.
    #if !defined(_CCCL_COMPILER_HIPRTC)
    cuda::std::mdspan m(std::move(ptr));
    #else
    cuda::std::mdspan m(cuda::std::move(ptr));
    #endif

    CHECK_MDSPAN(m, d);
  }
#endif

  return 0;
}

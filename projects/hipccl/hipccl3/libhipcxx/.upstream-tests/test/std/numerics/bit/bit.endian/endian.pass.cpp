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

// UNSUPPORTED: c++98, c++03

// enum class endian;
// <cuda/std/bit>

#include <cuda/std/bit>
// #include <cuda/std/cstring>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

#include "test_macros.h"

__host__ __device__
int main(int, char**)
{
  static_assert(cuda::std::is_enum<cuda::std::endian>::value, "");

  // Check that E is a scoped enum by checking for conversions.
  typedef cuda::std::underlying_type<cuda::std::endian>::type UT;
  static_assert(!cuda::std::is_convertible<cuda::std::endian, UT>::value, "");

  // test that the enumeration values exist
  static_assert(cuda::std::endian::little == cuda::std::endian::little, "");
  static_assert(cuda::std::endian::big == cuda::std::endian::big, "");
  static_assert(cuda::std::endian::native == cuda::std::endian::native, "");
  static_assert(cuda::std::endian::little != cuda::std::endian::big, "");

  //  Technically not required, but true on all existing machines
  static_assert(
    cuda::std::endian::native == cuda::std::endian::little || cuda::std::endian::native == cuda::std::endian::big, "");

  //  Try to check at runtime
  {
    uint32_t i = 0x01020304;
    char c[4];
    static_assert(sizeof(i) == sizeof(c), "");
    memcpy(c, &i, sizeof(c));

    assert((c[0] == 1) == (cuda::std::endian::native == cuda::std::endian::big));
  }

  return 0;
}

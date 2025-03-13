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

// test climits

#include <cuda/std/climits>

#include "test_macros.h"

#ifndef CHAR_BIT
#  error CHAR_BIT not defined
#endif

#ifndef SCHAR_MIN
#  error SCHAR_MIN not defined
#endif

#ifndef SCHAR_MAX
#  error SCHAR_MAX not defined
#endif

#ifndef UCHAR_MAX
#  error UCHAR_MAX not defined
#endif

#ifndef CHAR_MIN
#  error CHAR_MIN not defined
#endif

#ifndef CHAR_MAX
#  error CHAR_MAX not defined
#endif

// #ifndef MB_LEN_MAX
// #error MB_LEN_MAX not defined
// #endif

#ifndef SHRT_MIN
#  error SHRT_MIN not defined
#endif

#ifndef SHRT_MAX
#  error SHRT_MAX not defined
#endif

#ifndef USHRT_MAX
#  error USHRT_MAX not defined
#endif

#ifndef INT_MIN
#  error INT_MIN not defined
#endif

#ifndef INT_MAX
#  error INT_MAX not defined
#endif

#ifndef UINT_MAX
#  error UINT_MAX not defined
#endif

#ifndef LONG_MIN
#  error LONG_MIN not defined
#endif

#ifndef LONG_MAX
#  error LONG_MAX not defined
#endif

#ifndef ULONG_MAX
#  error ULONG_MAX not defined
#endif

#ifndef LLONG_MIN
#  error LLONG_MIN not defined
#endif

#ifndef LLONG_MAX
#  error LLONG_MAX not defined
#endif

#ifndef ULLONG_MAX
#  error ULLONG_MAX not defined
#endif

// test if __CHAR_UNSIGNED__ detection for NVRTC works correctly
// if not, go take a look at cuda/std/climits
#if TEST_COMPILER(NVRTC) || defined(TEST_COMPILER_HIPRTC)
#  include <cuda/std/type_traits>
static_assert(__CHAR_UNSIGNED__ == cuda::std::is_unsigned<char>::value, "");
#endif // TEST_COMPILER(NVRTC)

int main(int, char**)
{
  return 0;
}

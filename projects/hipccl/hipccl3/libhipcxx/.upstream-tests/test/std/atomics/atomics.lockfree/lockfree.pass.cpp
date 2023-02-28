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

//
// UNSUPPORTED: hipcc
// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

// <cuda/std/atomic>

// #define ATOMIC_BOOL_LOCK_FREE unspecified
// #define ATOMIC_CHAR_LOCK_FREE unspecified
// #define ATOMIC_CHAR16_T_LOCK_FREE unspecified
// #define ATOMIC_CHAR32_T_LOCK_FREE unspecified
// #define ATOMIC_WCHAR_T_LOCK_FREE unspecified
// #define ATOMIC_SHORT_LOCK_FREE unspecified
// #define ATOMIC_INT_LOCK_FREE unspecified
// #define ATOMIC_LONG_LOCK_FREE unspecified
// #define ATOMIC_LLONG_LOCK_FREE unspecified
// #define ATOMIC_POINTER_LOCK_FREE unspecified

#include <cuda/std/atomic>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
  assert(LIBCUDACXX_ATOMIC_BOOL_LOCK_FREE == 0 || LIBCUDACXX_ATOMIC_BOOL_LOCK_FREE == 1
         || LIBCUDACXX_ATOMIC_BOOL_LOCK_FREE == 2);
  assert(LIBCUDACXX_ATOMIC_CHAR_LOCK_FREE == 0 || LIBCUDACXX_ATOMIC_CHAR_LOCK_FREE == 1
         || LIBCUDACXX_ATOMIC_CHAR_LOCK_FREE == 2);
  assert(LIBCUDACXX_ATOMIC_CHAR16_T_LOCK_FREE == 0 || LIBCUDACXX_ATOMIC_CHAR16_T_LOCK_FREE == 1
         || LIBCUDACXX_ATOMIC_CHAR16_T_LOCK_FREE == 2);
  assert(LIBCUDACXX_ATOMIC_CHAR32_T_LOCK_FREE == 0 || LIBCUDACXX_ATOMIC_CHAR32_T_LOCK_FREE == 1
         || LIBCUDACXX_ATOMIC_CHAR32_T_LOCK_FREE == 2);
  assert(LIBCUDACXX_ATOMIC_WCHAR_T_LOCK_FREE == 0 || LIBCUDACXX_ATOMIC_WCHAR_T_LOCK_FREE == 1
         || LIBCUDACXX_ATOMIC_WCHAR_T_LOCK_FREE == 2);
  assert(LIBCUDACXX_ATOMIC_SHORT_LOCK_FREE == 0 || LIBCUDACXX_ATOMIC_SHORT_LOCK_FREE == 1
         || LIBCUDACXX_ATOMIC_SHORT_LOCK_FREE == 2);
  assert(LIBCUDACXX_ATOMIC_INT_LOCK_FREE == 0 || LIBCUDACXX_ATOMIC_INT_LOCK_FREE == 1
         || LIBCUDACXX_ATOMIC_INT_LOCK_FREE == 2);
  assert(LIBCUDACXX_ATOMIC_LONG_LOCK_FREE == 0 || LIBCUDACXX_ATOMIC_LONG_LOCK_FREE == 1
         || LIBCUDACXX_ATOMIC_LONG_LOCK_FREE == 2);
  assert(LIBCUDACXX_ATOMIC_LLONG_LOCK_FREE == 0 || LIBCUDACXX_ATOMIC_LLONG_LOCK_FREE == 1
         || LIBCUDACXX_ATOMIC_LLONG_LOCK_FREE == 2);
  assert(LIBCUDACXX_ATOMIC_POINTER_LOCK_FREE == 0 || LIBCUDACXX_ATOMIC_POINTER_LOCK_FREE == 1
         || LIBCUDACXX_ATOMIC_POINTER_LOCK_FREE == 2);

  return 0;
}

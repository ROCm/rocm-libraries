//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
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

#include <cuda/std/ctime>
#include <cuda/std/type_traits>

#include "test_macros.h"

#ifndef NULL
#  error NULL not defined
#endif

#if !defined(TEST_COMPILER_NVRTC) && !defined(TEST_COMPILER_HIPRTC)
#  ifndef CLOCKS_PER_SEC
#    error CLOCKS_PER_SEC not defined
#  endif
#endif

#if defined(TEST_HAS_C11_FEATURES)
#  ifndef TIME_UTC
#    error TIME_UTC not defined
#  endif
#endif

TEST_DIAG_SUPPRESS_GCC("-Wformat-zero-length")
TEST_NV_DIAG_SUPPRESS(set_but_not_used)

int main(int, char**)
{
  cuda::std::clock_t c = 0;
  cuda::std::size_t s  = 0;
  cuda::std::time_t t  = 0;
  unused(c); // Prevent unused warning
  unused(s); // Prevent unused warning
  unused(t); // Prevent unused warning
#if !defined(TEST_COMPILER_NVRTC) && !defined(TEST_COMPILER_HIPRTC)
  cuda::std::tm tm = {};
  char str[3];
  unused(tm); // Prevent unused warning
  unused(str); // Prevent unused warning
#  if defined(TEST_HAS_C11_FEATURES)
  cuda::std::timespec tmspec = {};
  unused(tmspec); // Prevent unused warning
#  endif // TEST_HAS_C11_FEATURES

//FIXME(HIP): clock() is declared as extern clock_t clock (void) __THROW; in <time.h>. clock_t is a typedef to long.
//The below test doesn't work for HIP, as the clock() function is defined in /opt/rocm/include/hip/amd_detail/amd_device_functions.h
//as "long long int  clock() { return __clock(); }". Therefore, the decltype expands to long long instead of the expected type long.
#  if !TEST_CUDA_COMPILER(CLANG) && !defined(__HIP__)
  static_assert((cuda::std::is_same<decltype(cuda::std::clock()), cuda::std::clock_t>::value), "");
#  endif // TEST_CUDA_COMPILER(CLANG)
  static_assert((cuda::std::is_same<decltype(cuda::std::difftime(t, t)), double>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::mktime(&tm)), cuda::std::time_t>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::time(&t)), cuda::std::time_t>::value), "");
#  if defined(TEST_HAS_TIMESPEC_GET)
  static_assert((cuda::std::is_same<decltype(cuda::std::timespec_get(&tmspec, 0)), int>::value), "");
#  endif // TEST_HAS_TIMESPEC_GET
#  ifndef _LIBCUDACXX_HAS_NO_THREAD_UNSAFE_C_FUNCTIONS
  static_assert((cuda::std::is_same<decltype(cuda::std::asctime(&tm)), char*>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::ctime(&t)), char*>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::gmtime(&t)), cuda::std::tm*>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::localtime(&t)), cuda::std::tm*>::value), "");
#  endif
  static_assert((cuda::std::is_same<decltype(cuda::std::strftime(str, s, "", &tm)), cuda::std::size_t>::value), "");
#endif

  return 0;
}

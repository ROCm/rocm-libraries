//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

// NOTE(HIP):
// This test is including hip/std/string which does not exist.
// HIPCC gives the correct error: hip/std/string file not found.
// However, other errors are expected, consider the expected-error and expected-note.
// We found _LIBCUDACXX_HAS_STRING guard in other tests that include the header,
// indicating that the header is not implemented yet.
#if defined(_LIBCUDACXX_HAS_STRING)
#   include <cuda/std/string>
#endif
#include <cuda/std/__memory_>
#include <cuda/std/tuple>

#include "test_macros.h"

struct UserType
{};

void test_bad_index()
{
  cuda::std::tuple<long, long, char,
  #if defined(_LIBCUDACXX_HAS_STRING)
    cuda::std::string,
  #endif
    char, UserType, char> t1;
  TEST_IGNORE_NODISCARD cuda::std::get<int>(t1); // expected-error@*:* {{type not found}}
  TEST_IGNORE_NODISCARD cuda::std::get<long>(t1); // expected-note {{requested here}}
  TEST_IGNORE_NODISCARD cuda::std::get<char>(t1); // expected-note {{requested here}}
                                                  // expected-error@*:* 2 {{type occurs more than once}}
  cuda::std::tuple<> t0;
  TEST_IGNORE_NODISCARD cuda::std::get<char*>(t0); // expected-node {{requested here}}
                                                   // expected-error@*:* 1 {{type not in empty type list}}
}

void test_bad_return_type()
{
  typedef cuda::std::unique_ptr<int> upint;
  cuda::std::tuple<upint> t;
  upint p = cuda::std::get<upint>(t); // expected-error {{deleted constructor}}
}

int main(int, char**)
{
  test_bad_index();
  test_bad_return_type();

  return 0;
}

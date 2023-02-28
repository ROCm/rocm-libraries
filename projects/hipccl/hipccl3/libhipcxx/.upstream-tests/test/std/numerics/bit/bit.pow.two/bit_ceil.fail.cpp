//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
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

// UNSUPPORTED: hipcc
// UNSUPPORTED: c++98, c++03

// template <class T>
//   constexpr T bit_ceil(T x) noexcept;

// Remarks: This function shall not participate in overload resolution unless
//	T is an unsigned integer type

#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include "test_macros.h"

class A
{};
enum E1 : unsigned char
{
  rEd
};
enum class E2 : unsigned char
{
  red
};

template <typename T>
__host__ __device__ constexpr bool toobig()
{
  return 0 == cuda::std::bit_ceil(cuda::std::numeric_limits<T>::max());
}

int main(int, char**)
{
  //	Make sure we generate a compile-time error for UB
  static_assert(toobig<unsigned char>(), ""); // expected-error-re {{{{(static_assert|static assertion)}} expression is
                                              // not an integral constant expression}}
  static_assert(toobig<unsigned short>(), ""); // expected-error-re {{{{(static_assert|static assertion)}} expression is
                                               // not an integral constant expression}}
  static_assert(toobig<unsigned>(), ""); // expected-error-re {{{{(static_assert|static assertion)}} expression is not
                                         // an integral constant expression}}
  static_assert(toobig<unsigned long>(), ""); // expected-error-re {{{{(static_assert|static assertion)}} expression is
                                              // not an integral constant expression}}
  static_assert(toobig<unsigned long long>(), ""); // expected-error-re {{{{(static_assert|static assertion)}}
                                                   // expression is not an integral constant expression}}

  static_assert(toobig<uint8_t>(), ""); // expected-error-re {{{{(static_assert|static assertion)}} expression is not an
                                        // integral constant expression}}
  static_assert(toobig<uint16_t>(), ""); // expected-error-re {{{{(static_assert|static assertion)}} expression is not
                                         // an integral constant expression}}
  static_assert(toobig<uint32_t>(), ""); // expected-error-re {{{{(static_assert|static assertion)}} expression is not
                                         // an integral constant expression}}
  static_assert(toobig<uint64_t>(), ""); // expected-error-re {{{{(static_assert|static assertion)}} expression is not
                                         // an integral constant expression}}
  static_assert(toobig<size_t>(), ""); // expected-error-re {{{{(static_assert|static assertion)}} expression is not an
                                       // integral constant expression}}
  static_assert(toobig<uintmax_t>(), ""); // expected-error-re {{{{(static_assert|static assertion)}} expression is not
                                          // an integral constant expression}}
  static_assert(toobig<uintptr_t>(), ""); // expected-error-re {{{{(static_assert|static assertion)}} expression is not
                                          // an integral constant expression}}

  return 0;
}

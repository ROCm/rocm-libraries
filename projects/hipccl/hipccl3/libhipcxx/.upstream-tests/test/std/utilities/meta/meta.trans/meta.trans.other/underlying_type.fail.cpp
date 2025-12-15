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

// type_traits

// underlying_type
// Mandates: enum must not be an incomplete enumeration type.

#include <cuda/std/climits>
#include <cuda/std/type_traits>

#include "test_macros.h"

  enum E1 {
    E1Zero,
    E1One,
    E1Two = sizeof(cuda::std::underlying_type<E1>::type)
  }; // expected-error@*:* {{cannot determine underlying type of incomplete enumeration type 'E1'}}

//  None of these are incomplete.
//  Scoped enums have an underlying type of 'int' unless otherwise specified
//  Unscoped enums with a specified underlying type become complete as soon as that type is specified.
// enum E2 : char            { E2Zero, E2One, E2Two = sizeof(cuda::std::underlying_type<E2>::type) };
// enum class E3             { E3Zero, E3One, E3Two = sizeof(cuda::std::underlying_type<E3>::type) };
// enum struct E4 : unsigned { E4Zero, E4One, E4Two = sizeof(cuda::std::underlying_type<E4>::type) };
// enum struct E5            { E5Zero, E5One, E5Two = sizeof(cuda::std::underlying_type<E5>::type) };
// enum class E6 : unsigned  { E6Zero, E6One, E6Two = sizeof(cuda::std::underlying_type<E6>::type) };

// These error messages will have to change if clang ever gets fixed. But at least they're being rejected.
enum E7 : cuda::std::underlying_type_t<E7>
{
}; // expected-error@*:* {{use of undeclared identifier 'E7'}}
enum class E8 : cuda::std::underlying_type_t<E8>
{
}; // expected-error@*:* {{use of undeclared identifier 'E8'}}
enum struct E9 : cuda::std::underlying_type_t<E9>
{
}; // expected-error@*:* {{use of undeclared identifier 'E9'}}

int main(int, char**)
{
  return 0;
}

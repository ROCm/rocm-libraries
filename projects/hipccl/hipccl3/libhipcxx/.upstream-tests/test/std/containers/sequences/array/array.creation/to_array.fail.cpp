//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
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

// <cuda/std/array>

#include <cuda/std/array>

#include "MoveOnly.h"
#include "test_macros.h"

// expected-warning@array:* 0-1 {{suggest braces around initialization of subobject}}

int main(int, char**)
{
  {
    char source[3][6] = {"hi", "world"};
    // expected-error-re@array:* {{to_array does not accept multidimensional arrays}}
    // expected-error-re@array:* {{to_array requires copy constructible elements}}
    // expected-error-re@array:* 3 {{cannot initialize}}
    cuda::std::to_array(source); // expected-note {{requested here}}
  }

  {
    MoveOnly mo[] = {MoveOnly{3}};
    // expected-error-re@array:* {{to_array requires copy constructible elements}}
    // expected-error-re@array:* {{{{(call to implicitly-deleted copy constructor of 'MoveOnly')|(call to deleted
    // constructor of 'MoveOnly')}}}}
    cuda::std::to_array(mo); // expected-note {{requested here}}
  }

  {
    const MoveOnly cmo[] = {MoveOnly{3}};
    // expected-error-re@array:* {{to_array requires move constructible elements}}
    // expected-error-re@array:* {{{{(call to implicitly-deleted copy constructor of 'MoveOnly')|(call to deleted
    // constructor of 'MoveOnly')}}}}
    cuda::std::to_array(cuda::std::move(cmo)); // expected-note {{requested here}}
  }

  return 0;
}

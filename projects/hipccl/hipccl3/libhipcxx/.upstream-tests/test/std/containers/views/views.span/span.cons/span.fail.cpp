//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
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


// <span>

// template<class OtherElementType, size_t OtherExtent>
//    constexpr span(const span<OtherElementType, OtherExtent>& s) noexcept;
//
//  Remarks: This constructor shall not participate in overload resolution unless:
//      Extent == dynamic_extent || Extent == OtherExtent is true, and
//      OtherElementType(*)[] is convertible to ElementType(*)[].

#include <cuda/std/cassert>
#include <cuda/std/span>

#include "test_macros.h"

template <class T, size_t extent, size_t otherExtent>
cuda::std::span<T, extent> createImplicitSpan(cuda::std::span<T, otherExtent> s)
{
  return {s}; // expected-error-re {{chosen constructor is explicit in copy-initialization}}
}

void checkCV()
{
  //  cuda::std::span<               int>   sp;
  cuda::std::span<const int> csp;
  cuda::std::span<volatile int> vsp;
  cuda::std::span<const volatile int> cvsp;

  //  cuda::std::span<               int, 0>   sp0;
  cuda::std::span<const int, 0> csp0;
  cuda::std::span<volatile int, 0> vsp0;
  cuda::std::span<const volatile int, 0> cvsp0;

  //  Try to remove const and/or volatile (dynamic -> dynamic)
  {
    cuda::std::span<int> s1{csp}; // expected-error {{no matching constructor for initialization of
                                  // 'cuda::std::span<int>'}}
    cuda::std::span<int> s2{vsp}; // expected-error {{no matching constructor for initialization of
                                  // 'cuda::std::span<int>'}}
    cuda::std::span<int> s3{cvsp}; // expected-error {{no matching constructor for initialization of
                                   // 'cuda::std::span<int>'}}

    cuda::std::span<const int> s4{vsp}; // expected-error {{no matching constructor for initialization of
                                        // 'cuda::std::span<const int>'}}
    cuda::std::span<const int> s5{cvsp}; // expected-error {{no matching constructor for initialization of
                                         // 'cuda::std::span<const int>'}}

    cuda::std::span<volatile int> s6{csp}; // expected-error {{no matching constructor for initialization of
                                           // 'cuda::std::span<volatile int>'}}
    cuda::std::span<volatile int> s7{cvsp}; // expected-error {{no matching constructor for initialization of
                                            // 'cuda::std::span<volatile int>'}}
  }

  //  Try to remove const and/or volatile (static -> static)
  {
    cuda::std::span<int, 0> s1{csp0}; // expected-error {{no matching constructor for initialization of
                                      // 'cuda::std::span<int, 0>'}}
    cuda::std::span<int, 0> s2{vsp0}; // expected-error {{no matching constructor for initialization of
                                      // 'cuda::std::span<int, 0>'}}
    cuda::std::span<int, 0> s3{cvsp0}; // expected-error {{no matching constructor for initialization of
                                       // 'cuda::std::span<int, 0>'}}

    cuda::std::span<const int, 0> s4{vsp0}; // expected-error {{no matching constructor for initialization of
                                            // 'cuda::std::span<const int, 0>'}}
    cuda::std::span<const int, 0> s5{cvsp0}; // expected-error {{no matching constructor for initialization of
                                             // 'cuda::std::span<const int, 0>'}}

    cuda::std::span<volatile int, 0> s6{csp0}; // expected-error {{no matching constructor for initialization of
                                               // 'cuda::std::span<volatile int, 0>'}}
    cuda::std::span<volatile int, 0> s7{cvsp0}; // expected-error {{no matching constructor for initialization of
                                                // 'cuda::std::span<volatile int, 0>'}}
  }

  //  Try to remove const and/or volatile (static -> dynamic)
  {
    cuda::std::span<int> s1{csp0}; // expected-error {{no matching constructor for initialization of
                                   // 'cuda::std::span<int>'}}
    cuda::std::span<int> s2{vsp0}; // expected-error {{no matching constructor for initialization of
                                   // 'cuda::std::span<int>'}}
    cuda::std::span<int> s3{cvsp0}; // expected-error {{no matching constructor for initialization of
                                    // 'cuda::std::span<int>'}}

    cuda::std::span<const int> s4{vsp0}; // expected-error {{no matching constructor for initialization of
                                         // 'cuda::std::span<const int>'}}
    cuda::std::span<const int> s5{cvsp0}; // expected-error {{no matching constructor for initialization of
                                          // 'cuda::std::span<const int>'}}

    cuda::std::span<volatile int> s6{csp0}; // expected-error {{no matching constructor for initialization of
                                            // 'cuda::std::span<volatile int>'}}
    cuda::std::span<volatile int> s7{cvsp0}; // expected-error {{no matching constructor for initialization of
                                             // 'cuda::std::span<volatile int>'}}
  }

  //  Try to remove const and/or volatile (static -> static)
  {
    cuda::std::span<int, 0> s1{csp}; // expected-error {{no matching constructor for initialization of
                                     // 'cuda::std::span<int, 0>'}}
    cuda::std::span<int, 0> s2{vsp}; // expected-error {{no matching constructor for initialization of
                                     // 'cuda::std::span<int, 0>'}}
    cuda::std::span<int, 0> s3{cvsp}; // expected-error {{no matching constructor for initialization of
                                      // 'cuda::std::span<int, 0>'}}

    cuda::std::span<const int, 0> s4{vsp}; // expected-error {{no matching constructor for initialization of
                                           // 'cuda::std::span<const int, 0>'}}
    cuda::std::span<const int, 0> s5{cvsp}; // expected-error {{no matching constructor for initialization of
                                            // 'cuda::std::span<const int, 0>'}}

    cuda::std::span<volatile int, 0> s6{csp}; // expected-error {{no matching constructor for initialization of
                                              // 'cuda::std::span<volatile int, 0>'}}
    cuda::std::span<volatile int, 0> s7{cvsp}; // expected-error {{no matching constructor for initialization of
                                               // 'cuda::std::span<volatile int, 0>'}}
  }
}

int main(int, char**)
{
  cuda::std::span<int> sp;
  cuda::std::span<int, 0> sp0;

  cuda::std::span<float> s1{sp}; // expected-error {{no matching constructor for initialization of
                                 // 'cuda::std::span<float>'}}
  cuda::std::span<float> s2{sp0}; // expected-error {{no matching constructor for initialization of
                                  // 'cuda::std::span<float>'}}
  cuda::std::span<float, 0> s3{sp}; // expected-error {{no matching constructor for initialization of
                                    // 'cuda::std::span<float, 0>'}}
  cuda::std::span<float, 0> s4{sp0}; // expected-error {{no matching constructor for initialization of
                                     // 'cuda::std::span<float, 0>'}}

  checkCV();

  // explicit constructor necessary
  {
    createImplicitSpan<int, 1>(sp);
  }

  return 0;
}

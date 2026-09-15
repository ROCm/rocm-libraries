//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
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

// <cuda/std/complex>

// Test that UDT's convertible to an integral or floating point type do not
// participate in overload resolution.

#include <cuda/std/cassert>
#include <cuda/std/complex>
#include <cuda/std/type_traits>

template <class IntT>
struct UDT
{
  operator IntT() const
  {
    return 1;
  }
};
#if defined(__HIP_PLATFORM_AMD__)
__host__ __device__ UDT<float> ft;
__host__ __device__ UDT<double> dt;
// CUDA treats long double as double
__host__ __device__ UDT<long double> ldt;
__host__ __device__ UDT<int> it;
__host__ __device__ UDT<unsigned long> uit;
#else
UDT<float> ft;
UDT<double> dt;
// CUDA treats long double as double
// UDT<long double> ldt;
UDT<int> it;
UDT<unsigned long> uit;
#endif

int main(int, char**)
{
  {
    cuda::std::real(ft); // expected-error {{no matching function}}
    cuda::std::real(dt); // expected-error {{no matching function}}
    #if defined(__HIP_PLATFORM_AMD__)
    cuda::std::real(ldt); // expected-error {{no matching function}}
    #endif
    cuda::std::real(it); // expected-error {{no matching function}}
    cuda::std::real(uit); // expected-error {{no matching function}}
  }
  {
    cuda::std::imag(ft); // expected-error {{no matching function}}
    cuda::std::imag(dt); // expected-error {{no matching function}}
    #if defined(__HIP_PLATFORM_AMD__)
    cuda::std::imag(ldt); // expected-error {{no matching function}}
    #endif
    cuda::std::imag(it); // expected-error {{no matching function}}
    cuda::std::imag(uit); // expected-error {{no matching function}}
  }
  {
    cuda::std::arg(ft); // expected-error {{no matching function}}
    cuda::std::arg(dt); // expected-error {{no matching function}}
    #if defined(__HIP_PLATFORM_AMD__)
    cuda::std::arg(ldt); // expected-error {{no matching function}}
    #endif
    cuda::std::arg(it); // expected-error {{no matching function}}
    cuda::std::arg(uit); // expected-error {{no matching function}}
  }
  {
    cuda::std::norm(ft); // expected-error {{no matching function}}
    cuda::std::norm(dt); // expected-error {{no matching function}}
    #if defined(__HIP_PLATFORM_AMD__)
    cuda::std::norm(ldt); // expected-error {{no matching function}}
    #endif
    cuda::std::norm(it); // expected-error {{no matching function}}
    cuda::std::norm(uit); // expected-error {{no matching function}}
  }
  {
    cuda::std::conj(ft); // expected-error {{no matching function}}
    cuda::std::conj(dt); // expected-error {{no matching function}}
    #if defined(__HIP_PLATFORM_AMD__)
    cuda::std::conj(ldt); // expected-error {{no matching function}}
    #endif
    cuda::std::conj(it); // expected-error {{no matching function}}
    cuda::std::conj(uit); // expected-error {{no matching function}}
  }
  {
    cuda::std::proj(ft); // expected-error {{no matching function}}
    cuda::std::proj(dt); // expected-error {{no matching function}}
    #if defined(__HIP_PLATFORM_AMD__)
    cuda::std::proj(ldt); // expected-error {{no matching function}}
    #endif
    cuda::std::proj(it); // expected-error {{no matching function}}
    cuda::std::proj(uit); // expected-error {{no matching function}}
  }

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// MIT License
//
// Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

// <cuda/std/numbers>

#include <cuda/std/numbers>
#include <cuda/std/type_traits>

#include <test_macros.h>

template <class ExpectedT, class T>
__host__ __device__ constexpr bool test_defined(const T& value)
{
  static_assert(cuda::std::is_same_v<ExpectedT, T>);

  const ExpectedT* addr = &value;
  unused(addr);

  return true;
}

template <class T>
__host__ __device__ constexpr bool test_type()
{
  test_defined<T>(cuda::std::numbers::e_v<T>);
  test_defined<T>(cuda::std::numbers::log2e_v<T>);
  test_defined<T>(cuda::std::numbers::log10e_v<T>);
  test_defined<T>(cuda::std::numbers::pi_v<T>);
  test_defined<T>(cuda::std::numbers::inv_pi_v<T>);
  test_defined<T>(cuda::std::numbers::inv_sqrtpi_v<T>);
  test_defined<T>(cuda::std::numbers::ln2_v<T>);
  test_defined<T>(cuda::std::numbers::ln10_v<T>);
  test_defined<T>(cuda::std::numbers::sqrt2_v<T>);
  test_defined<T>(cuda::std::numbers::sqrt3_v<T>);
  test_defined<T>(cuda::std::numbers::inv_sqrt3_v<T>);
  test_defined<T>(cuda::std::numbers::egamma_v<T>);
  test_defined<T>(cuda::std::numbers::phi_v<T>);

  return true;
}

__host__ __device__ constexpr bool test()
{
  test_defined<double>(cuda::std::numbers::e);
  test_defined<double>(cuda::std::numbers::log2e);
  test_defined<double>(cuda::std::numbers::log10e);
  test_defined<double>(cuda::std::numbers::pi);
  test_defined<double>(cuda::std::numbers::inv_pi);
  test_defined<double>(cuda::std::numbers::inv_sqrtpi);
  test_defined<double>(cuda::std::numbers::ln2);
  test_defined<double>(cuda::std::numbers::ln10);
  test_defined<double>(cuda::std::numbers::sqrt2);
  test_defined<double>(cuda::std::numbers::sqrt3);
  test_defined<double>(cuda::std::numbers::inv_sqrt3);
  test_defined<double>(cuda::std::numbers::egamma);
  test_defined<double>(cuda::std::numbers::phi);

  test_type<float>();
  test_type<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_type<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _LIBCUDACXX_HAS_NVFP16()
// NOTE(HIP/AMD): for ROCm 7.10 and earlier constexpression setting of __half values is not possible
#  if MINIMUM_ROCM_VERSION(7, 11, 0)
  test_type<__half>();
#  endif
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test_type<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()

  return true;
}

__global__ void test_kernel()
{
  test();
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}

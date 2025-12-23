//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

// <cmath>

// clang-format off
#include <disable_nvfp_conversions_and_operators.h>
// clang-format on

#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_isinf(const T pos, bool expected)
{
  _CCCL_ASSERT(cuda::std::isinf(pos) == expected, "Positive infinity not correct detected");

  if constexpr (cuda::std::numeric_limits<T>::is_signed)
  {
    T neg{};

    if constexpr (cuda::std::numeric_limits<T>::is_integer)
    {
      // handle integer overflow when negating the minimum value
      neg = (pos == cuda::std::numeric_limits<T>::min()) ? cuda::std::numeric_limits<T>::max() : -pos;
    }
    else if constexpr (cuda::std::is_floating_point_v<T>)
    {
      neg = -pos;
    }
    else // nvfp types
    {
      neg = cuda::std::copysign(pos, cuda::std::numeric_limits<T>::lowest());
    }

    _CCCL_ASSERT(cuda::std::isinf(neg) == expected, "Negative infinity not correct detected");
  }
}

template <class T>
__host__ __device__ constexpr void test_type()
{
  static_assert(cuda::std::is_same_v<bool, decltype(cuda::std::isinf(T{}))>);

  // __nv_fp8_e8m0 cannot represent 0
#if _CCCL_HAS_NVFP8_E8M0()
  if constexpr (!cuda::std::is_same_v<T, __nv_fp8_e8m0>)
#endif // _CCCL_HAS_NVFP8_E8M0
  {
    // TODO(HIP/AMD): remove this workaround once we can initialize a half with 0 as a constexpr
    if constexpr (cuda::std::is_same_v<T, __half>){
      test_isinf(__half{__half_raw{.x = 0}}, false);
    }
    else{
      test_isinf(T{}, false);
    }
  }
  test_isinf(cuda::std::numeric_limits<T>::min(), false);
  test_isinf(cuda::std::numeric_limits<T>::max(), false);

  if constexpr (cuda::std::numeric_limits<T>::has_infinity)
  {
    test_isinf(cuda::std::numeric_limits<T>::infinity(), true);
  }

  if constexpr (cuda::std::numeric_limits<T>::has_quiet_NaN)
  {
    test_isinf(cuda::std::numeric_limits<T>::quiet_NaN(), false);
  }

  if constexpr (cuda::std::numeric_limits<T>::has_signaling_NaN)
  {
    test_isinf(cuda::std::numeric_limits<T>::signaling_NaN(), false);
  }

  if constexpr (cuda::std::numeric_limits<T>::has_denorm)
  {
    test_isinf(cuda::std::numeric_limits<T>::denorm_min(), false);
  }
}

__host__ __device__ constexpr bool test()
{
  test_type<float>();
  test_type<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_type<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_NVFP16()
  test_type<__half>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test_type<__nv_bfloat16>();
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
  test_type<__nv_fp8_e4m3>();
#endif // _CCCL_HAS_NVFP8_E4M3
#if _CCCL_HAS_NVFP8_E5M2()
  test_type<__nv_fp8_e5m2>();
#endif // _CCCL_HAS_NVFP8_E5M2
#if _CCCL_HAS_NVFP8_E8M0()
  test_type<__nv_fp8_e8m0>();
#endif // _CCCL_HAS_NVFP8_E8M0
#if _CCCL_HAS_NVFP6_E2M3()
  test_type<__nv_fp6_e2m3>();
#endif // _CCCL_HAS_NVFP6_E2M3
#if _CCCL_HAS_NVFP6_E3M2()
  test_type<__nv_fp6_e3m2>();
#endif // _CCCL_HAS_NVFP6_E3M2
#if _CCCL_HAS_NVFP4_E2M1()
  test_type<__nv_fp4_e2m1>();
#endif // _CCCL_HAS_NVFP4_E2M1

  test_type<signed char>();
  test_type<unsigned char>();
  test_type<signed short>();
  test_type<unsigned short>();
  test_type<signed int>();
  test_type<unsigned int>();
  test_type<signed long>();
  test_type<unsigned long>();
  test_type<signed long long>();
  test_type<unsigned long long>();
#if _CCCL_HAS_INT128()
  test_type<__int128_t>();
  test_type<__uint128_t>();
#endif // _CCCL_HAS_INT128()

  return true;
}

int main(int, char**)
{
  test();
  assert(test());
  return 0;
}

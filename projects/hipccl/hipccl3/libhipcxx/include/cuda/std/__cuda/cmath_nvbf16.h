// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

#ifndef _LIBCUDACXX___CUDA_CMATH_NVBF16_H
#define _LIBCUDACXX___CUDA_CMATH_NVBF16_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if defined(_LIBCUDACXX_HAS_NVBF16)

#if defined(__HIP_PLATFORM_AMD__)
#  include <hip/hip_bf16.h>
#else
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wunused-function")
#  include <cuda_bf16.h>
_CCCL_DIAG_POP
#endif


#  include <cuda/std/cstdint>

#  include <nv/target>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// trigonometric functions
_LIBCUDACXX_HIDE_FROM_ABI __hip_bfloat16 sin(__hip_bfloat16 __v)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE_LIBHIPCXX, (return ::hsin(__v);), (return __float2bfloat16(::sinf(__bfloat162float(__v)));))
}

_LIBCUDACXX_HIDE_FROM_ABI __hip_bfloat16 sinh(__hip_bfloat16 __v)
{
  return __float2bfloat16(::sinhf(__bfloat162float(__v)));
}

_LIBCUDACXX_HIDE_FROM_ABI __hip_bfloat16 cos(__hip_bfloat16 __v)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE_LIBHIPCXX, (return ::hcos(__v);), (return __float2bfloat16(::cosf(__bfloat162float(__v)));))
}

_LIBCUDACXX_HIDE_FROM_ABI __hip_bfloat16 cosh(__hip_bfloat16 __v)
{
  return __float2bfloat16(::coshf(__bfloat162float(__v)));
}

_LIBCUDACXX_HIDE_FROM_ABI __hip_bfloat16 exp(__hip_bfloat16 __v)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE_LIBHIPCXX, (return ::hexp(__v);), (return __float2bfloat16(::expf(__bfloat162float(__v)));))
}

_LIBCUDACXX_HIDE_FROM_ABI __hip_bfloat16 hypot(__hip_bfloat16 __x, __hip_bfloat16 __y)
{
  return __float2bfloat16(::hypotf(__bfloat162float(__x), __bfloat162float(__y)));
}

_LIBCUDACXX_HIDE_FROM_ABI __hip_bfloat16 atan2(__hip_bfloat16 __x, __hip_bfloat16 __y)
{
  return __float2bfloat16(::atan2f(__bfloat162float(__x), __bfloat162float(__y)));
}

_LIBCUDACXX_HIDE_FROM_ABI __hip_bfloat16 log(__hip_bfloat16 __x)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE_LIBHIPCXX, (return ::hlog(__x);), (return __float2bfloat16(::logf(__bfloat162float(__x)));))
}

_LIBCUDACXX_HIDE_FROM_ABI __hip_bfloat16 sqrt(__hip_bfloat16 __x)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE_LIBHIPCXX, (return ::hsqrt(__x);), (return __float2bfloat16(::sqrtf(__bfloat162float(__x)));))
}

// floating point helper
_LIBCUDACXX_HIDE_FROM_ABI bool signbit(__hip_bfloat16 __v)
{
  return ::signbit(::__bfloat162float(__v));
}

_LIBCUDACXX_HIDE_FROM_ABI bool __constexpr_isnan(__hip_bfloat16 __x) noexcept
{
  return ::__hisnan(__x);
}

_LIBCUDACXX_HIDE_FROM_ABI bool isnan(__hip_bfloat16 __v)
{
  return __constexpr_isnan(__v);
}

_LIBCUDACXX_HIDE_FROM_ABI bool __constexpr_isinf(__hip_bfloat16 __x) noexcept
{
#  if _CCCL_STD_VER >= 2020 && defined(_CCCL_CUDACC_BELOW_12_3)
  // this is a workaround for nvbug 4362808
  return !::__hisnan(__x) && ::__hisnan(__x - __x);
#  else // ^^^ C++20 && below 12.3 ^^^ / vvv C++17 or 12.3+ vvv
  return ::__hisinf(__x) != 0;
#  endif // _CCCL_STD_VER <= 2017 || _CCCL_CUDACC_VER < 1203000
}

_LIBCUDACXX_HIDE_FROM_ABI bool isinf(__hip_bfloat16 __v)
{
  return __constexpr_isinf(__v);
}

_LIBCUDACXX_HIDE_FROM_ABI bool __constexpr_isfinite(__hip_bfloat16 __x) noexcept
{
  return !__constexpr_isnan(__x) && !__constexpr_isinf(__x);
}

_LIBCUDACXX_HIDE_FROM_ABI bool isfinite(__hip_bfloat16 __v)
{
  return __constexpr_isfinite(__v);
}

_LIBCUDACXX_HIDE_FROM_ABI __hip_bfloat16 __constexpr_copysign(__hip_bfloat16 __x, __hip_bfloat16 __y) noexcept
{
  return __float2bfloat16(::copysignf(__bfloat162float(__x), __bfloat162float(__y)));
}

_LIBCUDACXX_HIDE_FROM_ABI __hip_bfloat16 copysign(__hip_bfloat16 __x, __hip_bfloat16 __y)
{
  return __constexpr_copysign(__x, __y);
}

_LIBCUDACXX_HIDE_FROM_ABI __hip_bfloat16 __constexpr_fabs(__hip_bfloat16 __x) noexcept
{
  return ::__habs(__x);
}

_LIBCUDACXX_HIDE_FROM_ABI __hip_bfloat16 fabs(__hip_bfloat16 __x)
{
  return __constexpr_fabs(__x);
}

_LIBCUDACXX_HIDE_FROM_ABI __hip_bfloat16 abs(__hip_bfloat16 __x)
{
  return __constexpr_fabs(__x);
}

_LIBCUDACXX_HIDE_FROM_ABI __hip_bfloat16 __constexpr_fmax(__hip_bfloat16 __x, __hip_bfloat16 __y) noexcept
{
#ifndef __HIP_DEVICE_COMPILE__
  // NOTE(HIP/AMD): currently __hmax might return incorrect results if one of the variables is NaN
  // for host calculations. Especially for the combination __hmax(nan,inf) it return nan instead of inf!
  if(__constexpr_isnan(__x)) return __y;
  if(__constexpr_isnan(__y)) return __x;
#endif
  return ::__hmax(__x, __y);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif /// _LIBCUDACXX_HAS_NVBF16

#endif // _LIBCUDACXX___CUDA_CMATH_NVBF16_H

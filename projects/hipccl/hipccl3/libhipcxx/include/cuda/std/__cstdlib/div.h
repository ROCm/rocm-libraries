// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

#ifndef _LIBCUDACXX___CSTDLIB_DIV_H
#define _LIBCUDACXX___CSTDLIB_DIV_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC) && !defined(_CCCL_COMPILER_HIPRTC)
#  include <cstdlib>
#endif // !_CCCL_COMPILER(NVRTC)

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// If available, use the host's div_t, ldiv_t, and lldiv_t types because the struct members order is
// implementation-defined.
#if !_CCCL_COMPILER(NVRTC) && !defined(_CCCL_COMPILER_HIPRTC)
using ::div_t;
using ::ldiv_t;
using ::lldiv_t;
#else // ^^^ !_CCCL_COMPILER(NVRTC) / _CCCL_COMPILER(NVRTC) vvv
struct _CCCL_TYPE_VISIBILITY_DEFAULT div_t
{
  int quot;
  int rem;
};

struct _CCCL_TYPE_VISIBILITY_DEFAULT ldiv_t
{
  long quot;
  long rem;
};

struct _CCCL_TYPE_VISIBILITY_DEFAULT lldiv_t
{
  long long quot;
  long long rem;
};
#endif // !_CCCL_COMPILER(NVRTC)

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr div_t div(int __x, int __y) noexcept
{
  div_t __result{};
  __result.quot = __x / __y;
  __result.rem  = __x % __y;
  return __result;
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr ldiv_t ldiv(long __x, long __y) noexcept
{
  ldiv_t __result{};
  __result.quot = __x / __y;
  __result.rem  = __x % __y;
  return __result;
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr ldiv_t div(long __x, long __y) noexcept
{
  return _CUDA_VSTD::ldiv(__x, __y);
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr lldiv_t lldiv(long long __x, long long __y) noexcept
{
  lldiv_t __result{};
  __result.quot = __x / __y;
  __result.rem  = __x % __y;
  return __result;
}

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr lldiv_t div(long long __x, long long __y) noexcept
{
  return _CUDA_VSTD::lldiv(__x, __y);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CSTDLIB_DIV_H

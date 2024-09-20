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

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_EXTENDED_FLOATING_POINT_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_EXTENDED_FLOATING_POINT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
struct __is_extended_floating_point : false_type
{};

#if _CCCL_STD_VER >= 2017 && defined(__cpp_inline_variables) && (__cpp_inline_variables >= 201606L)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool __is_extended_floating_point_v = false;
#elif _CCCL_STD_VER >= 2014 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool __is_extended_floating_point_v = __is_extended_floating_point<_Tp>::value;
#endif // _CCCL_STD_VER >= 2014

#if defined(_LIBCUDACXX_HAS_NVFP16)

#if defined(__HIP_PLATFORM_AMD__)
#  include <hip/hip_fp16.h>
#else
#  include <cuda_fp16.h>
#endif

template <>
struct __is_extended_floating_point<__half> : true_type
{};

#  if _CCCL_STD_VER >= 2017 && defined(__cpp_inline_variables) && (__cpp_inline_variables >= 201606L)
template <>
_LIBCUDACXX_INLINE_VAR constexpr bool __is_extended_floating_point_v<__half> = true;
#  endif // _CCCL_STD_VER >= 2014
#endif // _LIBCUDACXX_HAS_NVFP16

#if defined(_LIBCUDACXX_HAS_NVBF16)
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wunused-function")
#if defined(__HIP_PLATFORM_AMD__)
#  include <hip/hip_bf16.h>
#else
#  include <cuda_bf16.h>
#endif
_CCCL_DIAG_POP

template <>
struct __is_extended_floating_point<__hip_bfloat16> : true_type
{};

#  if _CCCL_STD_VER >= 2017 && defined(__cpp_inline_variables) && (__cpp_inline_variables >= 201606L)
template <>
_LIBCUDACXX_INLINE_VAR constexpr bool __is_extended_floating_point_v<__hip_bfloat16> = true;
#  endif // _CCCL_STD_VER >= 2014
#endif // _LIBCUDACXX_HAS_NVBF16

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_EXTENDED_FLOATING_POINT_H

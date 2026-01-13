// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
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

#ifndef _LIBCUDACXX___CSTDDEF_TYPES_H
#define _LIBCUDACXX___CSTDDEF_TYPES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC) && !defined(_CCCL_COMPILER_HIPRTC)
#  include <cstddef>
#else
#  if !defined(offsetof)
// NOTE(HIP/AMD): C++ does not allow accessing a member through a null pointer in a constant expression.
// The following is a true constant expression and does not dereference a null pointer.
#  if defined(_CCCL_COMPILER_HIPRTC)
#    define offsetof(type, member) __builtin_offsetof(type, member)
#else
#    define offsetof(type, member) (::size_t) ((char*) &(((type*) 0)->member) - (char*) 0)
#endif
#  endif // !offsetof
#endif // !_CCCL_COMPILER(NVRTC)

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(__CLANG_MAX_ALIGN_T_DEFINED) || defined(_GCC_MAX_ALIGN_T) || defined(__DEFINED_max_align_t) \
  || defined(__NetBS)
// Re-use the compiler's <stddef.h> max_align_t where possible.
using ::max_align_t;
#else
using max_align_t = long double;
#endif

using nullptr_t = decltype(nullptr);
using ::ptrdiff_t;
using ::size_t;

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CSTDDEF_TYPES_H

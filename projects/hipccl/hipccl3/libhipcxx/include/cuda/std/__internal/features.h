//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===---------------------------------------------------------------------===//

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

#ifndef _LIBCUDACXX___INTERNAL_FEATURES_H
#define _LIBCUDACXX___INTERNAL_FEATURES_H

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#define _LIBCUDACXX_HAS_CXX20_CHRONO_LITERALS() 0
#define _LIBCUDACXX_HAS_EXTERNAL_ATOMIC_IMP()   1
#define _LIBCUDACXX_HAS_MONOTONIC_CLOCK()       0
#define _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()    0
#define _LIBCUDACXX_HAS_WCHAR_H()               0

#if _CCCL_HAS_CUDA_COMPILER() || __cpp_aligned_new < 201606
#  define _LIBCUDACXX_HAS_ALIGNED_ALLOCATION() 0
#else
#  define _LIBCUDACXX_HAS_ALIGNED_ALLOCATION() 1
#endif // !_CCCL_HAS_CUDA_COMPILER() && __cpp_aligned_new >= 201606

#if _CCCL_STD_VER <= 2017 || !defined(__cpp_char8_t)
#  define _LIBCUDACXX_HAS_CHAR8_T() 0
#else
#  define _LIBCUDACXX_HAS_CHAR8_T() 1
#endif // _CCCL_STD_VER <= 2017 || !defined(__cpp_char8_t)

// We need `is_constant_evaluated` for clang and gcc. MSVC also needs extensive rework
#if !defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
#  define _LIBCUDACXX_HAS_CONSTEXPR_COMPLEX_OPERATIONS() 0
#elif _CCCL_COMPILER(NVRTC)
#  define _LIBCUDACXX_HAS_CONSTEXPR_COMPLEX_OPERATIONS() 0
#elif defined(_CCCL_COMPILER_HIPRTC)
#  define _LIBCUDACXX_HAS_CONSTEXPR_COMPLEX_OPERATIONS() 0
#elif _CCCL_COMPILER(MSVC)
#  define _LIBCUDACXX_HAS_CONSTEXPR_COMPLEX_OPERATIONS() 0
#elif _CCCL_CUDA_COMPILER(CLANG)
#  define _LIBCUDACXX_HAS_CONSTEXPR_COMPLEX_OPERATIONS() 0
#else
#  define _LIBCUDACXX_HAS_CONSTEXPR_COMPLEX_OPERATIONS() 1
#endif

#ifndef _LIBCUDACXX_HAS_NO_INCOMPLETE_RANGES
#  define _LIBCUDACXX_HAS_NO_INCOMPLETE_RANGES
#endif // _LIBCUDACXX_HAS_NO_INCOMPLETE_RANGES

// libcu++ requires host device support for its tests. Until then restrict usage to at least 12.2
#if _CCCL_HAS_NVFP16() && _CCCL_CTK_AT_LEAST(12, 2) \
  && (_CCCL_HAS_CUDA_COMPILER() || defined(LIBCUDACXX_ENABLE_HOST_NVFP16)) || defined(__HIP_PLATFORM_AMD__)
#  define _LIBCUDACXX_HAS_NVFP16() 1
#else
#  define _LIBCUDACXX_HAS_NVFP16() 0
#endif // _CCCL_HAS_NVFP16() && _CCCL_CTK_AT_LEAST(12, 2)

// libcu++ requires host device support for its tests. Until then restrict usage to at least 12.2
#if _CCCL_HAS_NVBF16() && _CCCL_CTK_AT_LEAST(12, 2) || defined(__HIP_PLATFORM_AMD__)
#  define _LIBCUDACXX_HAS_NVBF16() 1
#else
#  define _LIBCUDACXX_HAS_NVBF16() 0
#endif // _CCCL_HAS_NVBF16() && _CCCL_CTK_AT_LEAST(12, 2)

// NVCC does not have a way of silencing non '_' prefixed UDLs
#if !_CCCL_CUDA_COMPILER(NVCC) && !_CCCL_COMPILER(NVRTC) && !defined(_CCCL_COMPILER_HIPCC) && !defined(_CCCL_COMPILER_HIPRTC)
#  define _LIBCUDACXX_HAS_STL_LITERALS
#endif // !_CCCL_CUDA_COMPILER(NVCC) && !_CCCL_COMPILER(NVRTC)

#endif // _LIBCUDACXX___INTERNAL_FEATURES_H

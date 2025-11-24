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

#ifndef _LIBCUDACXX___FLOATING_POINT_NVFP_TYPES_H
#define _LIBCUDACXX___FLOATING_POINT_NVFP_TYPES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// Prevent resetting of the diagnostic state by guarding the push/pop with a macro
#if _CCCL_HAS_NVFP16()
#ifdef __HIP_PLATFORM_AMD__
#  include <hip/hip_fp16.h>
#else
_CCCL_DIAG_PUSH
#  include <cuda_fp16.h>
_CCCL_DIAG_POP
#endif
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
#ifdef __HIP_PLATFORM_AMD__
#  include <hip/hip_bf16.h>
#else
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wunused-function")
#  include <cuda_bf16.h>
_CCCL_DIAG_POP
#endif
#endif // _CCCL_HAS_NVBF16()

#if _CCCL_HAS_NVFP8()
#ifdef __HIP_PLATFORM_AMD__
#  include <hip/hip_fp8.h>
#else
_CCCL_DIAG_PUSH
#  include <cuda_fp8.h>
_CCCL_DIAG_POP
#endif
#endif // _CCCL_HAS_NVFP8()

#if _CCCL_HAS_NVFP6()
#ifdef __HIP_PLATFORM_AMD__
#  include <hip/hip_fp6.h>
#else
_CCCL_DIAG_PUSH
#  include <cuda_fp6.h>
_CCCL_DIAG_POP
#endif
#endif // _CCCL_HAS_NVFP6()

#if _CCCL_HAS_NVFP4()
#ifdef __HIP_PLATFORM_AMD__
#  include <hip/hip_fp4.h>
#else
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wunused-parameter")
_CCCL_DIAG_SUPPRESS_MSVC(4100) // unreferenced formal parameter
#  include <cuda_fp4.h>
_CCCL_DIAG_POP
#endif
#endif // _CCCL_HAS_NVFP4()

#endif // _LIBCUDACXX___FLOATING_POINT_NVFP_TYPES_H

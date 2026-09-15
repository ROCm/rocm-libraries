//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
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

#ifndef __CCCL_EXECUTION_SPACE_H
#define __CCCL_EXECUTION_SPACE_H

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILATION() || defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC_RTC__)
#  define _CCCL_HOST        __host__
#  define _CCCL_DEVICE      __device__
#  define _CCCL_HOST_DEVICE __host__ __device__
#else // ^^^ _CCCL_CUDA_COMPILATION ^^^ / vvv !_CCCL_CUDA_COMPILATION vvv
#  define _CCCL_HOST
#  define _CCCL_DEVICE
#  define _CCCL_HOST_DEVICE
#endif // !_CCCL_CUDA_COMPILATION

// Global variables of non builtin types are only device accessible if they are marked as `__device__`
#if _CCCL_DEVICE_COMPILATION() && !_CCCL_CUDA_COMPILER(NVHPC)
#  define _CCCL_GLOBAL_VARIABLE _CCCL_DEVICE
#else // ^^^ _CCCL_DEVICE_COMPILATION() && !_CCCL_CUDA_COMPILER(NVHPC) ^^^ /
      // vvv !_CCCL_DEVICE_COMPILATION() || _CCCL_CUDA_COMPILER(NVHPC) vvv
#  define _CCCL_GLOBAL_VARIABLE
#endif // ^^^ !_CCCL_DEVICE_COMPILATION() || _CCCL_CUDA_COMPILER(NVHPC) ^^^

/// In device code, _CCCL_PTX_ARCH() expands to the PTX version for which we are compiling.
/// In host code, _CCCL_PTX_ARCH()'s value is implementation defined.
#if !defined(__CUDA_ARCH__)
#  define _CCCL_PTX_ARCH() 0
#else
#  define _CCCL_PTX_ARCH() __CUDA_ARCH__
#endif

// Compile with NVCC compiler and only device code, Volta+  GPUs
#if _CCCL_CUDA_COMPILER(NVCC) && _CCCL_PTX_ARCH() >= 700
#  define _CCCL_GRID_CONSTANT __grid_constant__
#else // ^^^ _CCCL_CUDA_COMPILER(NVCC) ^^^ / vvv !_CCCL_CUDA_COMPILER(NVCC) vvv
#  define _CCCL_GRID_CONSTANT
#endif // _CCCL_CUDA_COMPILER(NVCC) && _CCCL_PTX_ARCH() >= 700

#if !defined(_CCCL_EXEC_CHECK_DISABLE)
#  if _CCCL_CUDA_COMPILER(NVCC)
#    define _CCCL_EXEC_CHECK_DISABLE _CCCL_PRAGMA(nv_exec_check_disable)
#  else
#    define _CCCL_EXEC_CHECK_DISABLE
#  endif // _CCCL_CUDA_COMPILER(NVCC)
#endif // !_CCCL_EXEC_CHECK_DISABLE

#endif // __CCCL_EXECUTION_SPACE_H

// -*- C++ -*-
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

#ifndef _LIBCUDACXX___CSTDLIB_MALLOC_H
#define _LIBCUDACXX___CSTDLIB_MALLOC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/cstring>

#if !_CCCL_COMPILER(NVRTC) && !defined(_CCCL_COMPILER_HIPRTC)
#  include <cstdlib>
#endif // !_CCCL_COMPILER(NVRTC)

#include <nv/target>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

using ::free;
using ::malloc;

// We need to ensure that we not only compile with a cuda compiler but also compile cuda source files
#if _CCCL_HAS_CUDA_COMPILER() && (defined(__CUDACC__) || defined(_NVHPC_CUDA) || defined(__HIPCC__))
_CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE void* __calloc_device(size_t __n, size_t __size) noexcept
{
  void* __ptr{};

  const size_t __nbytes = __n * __size;

  if (::__umul64hi(__n, __size) == 0)
  {
    __ptr = _CUDA_VSTD::malloc(__nbytes);
    if (__ptr != nullptr)
    {
      _CUDA_VSTD::memset(__ptr, 0, __nbytes);
    }
  }

  return __ptr;
}
#endif // _CCCL_HAS_CUDA_COMPILER() && (defined(__CUDACC__) || defined(_NVHPC_CUDA))

_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI void* calloc(size_t __n, size_t __size) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_HOST_LIBHIPCXX, (return ::calloc(__n, __size);), (return _CUDA_VSTD::__calloc_device(__n, __size);))
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CSTDLIB_MALLOC_H

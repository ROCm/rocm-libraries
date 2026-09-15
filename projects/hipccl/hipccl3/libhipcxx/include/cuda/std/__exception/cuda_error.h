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

#ifndef _LIBCUDACXX___EXCEPTION_CUDA_ERROR_H
#define _LIBCUDACXX___EXCEPTION_CUDA_ERROR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__exception/terminate.h>

#if !_CCCL_COMPILER(NVRTC) && !defined(_CCCL_COMPILER_HIPRTC)
#  include <cstdio>
#  include <stdexcept>
#endif // !_CCCL_COMPILER(NVRTC)

#include <nv/target>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

/**
 * @brief Exception thrown when a CUDA error is encountered.
 */
#ifndef _CCCL_NO_EXCEPTIONS
class cuda_error : public ::std::runtime_error
{
private:
  struct __msg_storage
  {
    char __buffer[256];
  };

  static char* __format_cuda_error(const int __status, const char* __msg, char* __msg_buffer) noexcept
  {
    ::snprintf(__msg_buffer, 256, "cudaError %d: %s", __status, __msg);
    return __msg_buffer;
  }

public:
  cuda_error(const int __status, const char* __msg, __msg_storage __msg_buffer = {0}) noexcept
      : ::std::runtime_error(__format_cuda_error(__status, __msg, __msg_buffer.__buffer))
  {}
};

_CCCL_NORETURN _LIBCUDACXX_HIDE_FROM_ABI void __throw_cuda_error(const int __status, const char* __msg)
{
  NV_IF_ELSE_TARGET(NV_IS_HOST_LIBHIPCXX,
                    (throw ::cuda::cuda_error(__status, __msg);),
                    ((void) __status; (void) __msg; _CUDA_VSTD_NOVERSION::terminate();))
}
#else // ^^^ !_CCCL_NO_EXCEPTIONS ^^^ / vvv _CCCL_NO_EXCEPTIONS vvv
class cuda_error
{
public:
  _LIBCUDACXX_HIDE_FROM_ABI cuda_error(const int, const char*) noexcept {}
};

_CCCL_NORETURN _LIBCUDACXX_HIDE_FROM_ABI void __throw_cuda_error(const int, const char*)
{
  _CUDA_VSTD_NOVERSION::terminate();
}
#endif // _CCCL_NO_EXCEPTIONS

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _LIBCUDACXX___EXCEPTION_CUDA_ERROR_H

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

#ifndef _CUDA__STD__CUDA_ENSURE_CURRENT_DEVICE_H
#define _CUDA__STD__CUDA_ENSURE_CURRENT_DEVICE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILER(CLANG) && !defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC_RTC__)
#  include <cuda_runtime_api.h>
#endif // _CCCL_CUDA_COMPILER(CLANG)

#include <cuda/std/__cuda/api_wrapper.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! @brief `__ensure_current_device` is a simple helper that the current device is set to the right one.
//! Only changes the current device if the target device is not the current one
struct __ensure_current_device
{
  int __target_device_   = 0;
  int __original_device_ = 0;

  //! @brief Queries the current device and if that is different than \p __target_device sets the current device to
  //! \p __target_device
  __ensure_current_device(const int __target_device)
      : __target_device_(__target_device)
  {
    _CCCL_TRY_CUDA_API(::hipGetDevice, "Failed to query current device", &__original_device_);
    if (__original_device_ != __target_device_)
    {
      _CCCL_TRY_CUDA_API(::hipSetDevice, "Failed to set device", __target_device_);
    }
  }

  //! @brief If the \p __original_device was not equal to \p __target_device sets the current device back to
  //! \p __original_device
  ~__ensure_current_device()
  {
    if (__original_device_ != __target_device_)
    {
      _CCCL_TRY_CUDA_API(::hipSetDevice, "Failed to set device", __original_device_);
    }
  }
};

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif //_CUDA__STD__CUDA_ENSURE_CURRENT_DEVICE_H

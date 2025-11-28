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

#ifndef _CUDA___BIT_BITMASK_H
#define _CUDA___BIT_BITMASK_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

//#include <cuda/__ptx/instructions/bmsk.h>
//#include <cuda/__ptx/instructions/shl.h>
//#include <cuda/__ptx/instructions/shr.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/limits>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <typename _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp __shl(const _Tp __value, int __shift) noexcept
{
  // Note(HIP/AMD): bfind uses ptx which is not available for AMD hardware
  /*
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    if constexpr (sizeof(_Tp) <= sizeof(uint64_t))
    {
      NV_DISPATCH_TARGET(NV_IS_DEVICE_LIBHIPCXX,
                         (using _Up = _CUDA_VSTD::_If<sizeof(_Tp) <= sizeof(uint32_t), uint32_t, uint64_t>;
                          return _CUDA_VPTX::shl(static_cast<_Up>(__value), __shift);))
    }
  }
  */
  return (__shift >= _CUDA_VSTD::numeric_limits<_Tp>::digits) ? _Tp{0} : __value << __shift;
}

template <typename _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp __shr(const _Tp __value, int __shift) noexcept
{
  // Note(HIP/AMD): bfind uses ptx which is not available for AMD hardware
  /*
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    if constexpr (sizeof(_Tp) <= sizeof(uint64_t))
    {
      NV_DISPATCH_TARGET(NV_IS_DEVICE_LIBHIPCXX,
                         (using _Up = _CUDA_VSTD::_If<sizeof(_Tp) <= sizeof(uint32_t), uint32_t, uint64_t>;
                          return _CUDA_VPTX::shr(static_cast<_Up>(__value), __shift);))
    }
  }
  */
  return (__shift >= _CUDA_VSTD::numeric_limits<_Tp>::digits) ? _Tp{0} : __value >> __shift;
}

template <typename _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp bitmask(int __start, int __width) noexcept
{
  static_assert(_CUDA_VSTD::__cccl_is_unsigned_integer_v<_Tp>, "bitmask() requires unsigned integer types");
  [[maybe_unused]] constexpr auto __digits = _CUDA_VSTD::numeric_limits<_Tp>::digits;
  _CCCL_ASSERT(__width >= 0 && __width <= __digits, "width out of range");
  _CCCL_ASSERT(__start >= 0 && __start <= __digits, "start position out of range");
  _CCCL_ASSERT(__start + __width <= __digits, "start position + width out of range");
  // Note(HIP/AMD): bfind uses ptx which is not available for AMD hardware
  /*
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    if constexpr (sizeof(_Tp) <= sizeof(uint32_t))
    {
      NV_IF_TARGET_LIBHIPCXX(NV_PROVIDES_SM_70, (return _CUDA_VPTX::bmsk_clamp(__start, __width);))
    }
  }
  */
  return ::cuda::__shl(static_cast<_Tp>(::cuda::__shl(_Tp{1}, __width) - 1), __start);
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___BIT_BITMASK_H

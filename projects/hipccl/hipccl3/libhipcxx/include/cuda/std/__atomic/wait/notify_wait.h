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

#ifndef _LIBCUDACXX___ATOMIC_WAIT_NOTIFY_WAIT_H
#define _LIBCUDACXX___ATOMIC_WAIT_NOTIFY_WAIT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__atomic/order.h>
#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/__atomic/wait/polling.h>
#include <cuda/std/cstring>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_DEVICE inline void __atomic_try_wait_unsupported_before_SM_70__(){};

template <typename _Tp, typename _Sco>
_LIBCUDACXX_HIDE_FROM_ABI void
__atomic_try_wait_slow(_Tp const volatile* __a, __atomic_underlying_remove_cv_t<_Tp> __val, memory_order __order, _Sco)
{
  #ifdef __HIP_PLATFORM_AMD__
  __atomic_try_wait_slow_fallback(__a, __val, __order, _Sco{});
  #else
  NV_DISPATCH_TARGET(NV_PROVIDES_SM_70, __atomic_try_wait_slow_fallback(__a, __val, __order, _Sco{});
                     , NV_IS_HOST_LIBHIPCXX, __atomic_try_wait_slow_fallback(__a, __val, __order, _Sco{});
                     , NV_ANY_TARGET_LIBHIPCXX, __atomic_try_wait_unsupported_before_SM_70__(););
  #endif
}

template <typename _Tp, typename _Sco>
_LIBCUDACXX_HIDE_FROM_ABI void __atomic_notify_one(_Tp const volatile*, _Sco)
{
  NV_DISPATCH_TARGET(NV_PROVIDES_SM_70, , NV_IS_HOST_LIBHIPCXX, , NV_ANY_TARGET_LIBHIPCXX, __atomic_try_wait_unsupported_before_SM_70__(););
}

template <typename _Tp, typename _Sco>
_LIBCUDACXX_HIDE_FROM_ABI void __atomic_notify_all(_Tp const volatile*, _Sco)
{
  NV_DISPATCH_TARGET(NV_PROVIDES_SM_70, , NV_IS_HOST_LIBHIPCXX, , NV_ANY_TARGET_LIBHIPCXX, __atomic_try_wait_unsupported_before_SM_70__(););
}

template <typename _Tp>
_LIBCUDACXX_HIDE_FROM_ABI bool __nonatomic_compare_equal(_Tp const& __lhs, _Tp const& __rhs)
{
#if _CCCL_HAS_CUDA_COMPILER() || defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC_RTC__)
  return __lhs == __rhs;
#else
  return _CUDA_VSTD::memcmp(&__lhs, &__rhs, sizeof(_Tp)) == 0;
#endif
}

template <typename _Tp, typename _Sco>
_LIBCUDACXX_HIDE_FROM_ABI void __atomic_wait(
  _Tp const volatile* __a, __atomic_underlying_remove_cv_t<_Tp> const __val, memory_order __order, _Sco = {})
{
  for (int __i = 0; __i < _LIBCUDACXX_POLLING_COUNT; ++__i)
  {
    if (!__nonatomic_compare_equal(__atomic_load_dispatch(__a, __order, _Sco{}), __val))
    {
      return;
    }
    if (__i < 12)
    {
      _CUDA_VSTD::__cccl_thread_yield_processor();
    }
    else
    {
      _CUDA_VSTD::__cccl_thread_yield();
    }
  }
  while (__nonatomic_compare_equal(__atomic_load_dispatch(__a, __order, _Sco{}), __val))
  {
    __atomic_try_wait_slow(__a, __val, __order, _Sco{});
  }
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ATOMIC_WAIT_NOTIFY_WAIT_H

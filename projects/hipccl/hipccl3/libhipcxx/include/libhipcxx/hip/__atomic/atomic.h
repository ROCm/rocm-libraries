// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
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

#ifndef _CUDA___ATOMIC_ATOMIC_H
#define _CUDA___ATOMIC_ATOMIC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/atomic>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

// atomic<T>

template <class _Tp, thread_scope _Sco = thread_scope::thread_scope_system>
struct atomic : public _CUDA_VSTD::__atomic_impl<_Tp, _Sco>
{
  using value_type = _Tp;

  _CCCL_HIDE_FROM_ABI constexpr atomic() noexcept = default;

  _LIBCUDACXX_HIDE_FROM_ABI constexpr atomic(_Tp __d) noexcept
      : _CUDA_VSTD::__atomic_impl<_Tp, _Sco>(__d)
  {}

  atomic(const atomic&)                     = delete;
  atomic& operator=(const atomic&)          = delete;
  atomic& operator=(const atomic&) volatile = delete;

  _LIBCUDACXX_HIDE_FROM_ABI _Tp operator=(_Tp __d) volatile noexcept
  {
    this->store(__d);
    return __d;
  }
  _LIBCUDACXX_HIDE_FROM_ABI _Tp operator=(_Tp __d) noexcept
  {
    this->store(__d);
    return __d;
  }

  _LIBCUDACXX_HIDE_FROM_ABI _Tp fetch_max(const _Tp& __op, memory_order __m = memory_order_seq_cst) noexcept
  {
    return _CUDA_VSTD::__atomic_fetch_max_dispatch(&this->__a, __op, __m, _CUDA_VSTD::__scope_to_tag<_Sco>{});
  }
  _LIBCUDACXX_HIDE_FROM_ABI _Tp fetch_max(const _Tp& __op, memory_order __m = memory_order_seq_cst) volatile noexcept
  {
    return _CUDA_VSTD::__atomic_fetch_max_dispatch(&this->__a, __op, __m, _CUDA_VSTD::__scope_to_tag<_Sco>{});
  }

  _LIBCUDACXX_HIDE_FROM_ABI _Tp fetch_min(const _Tp& __op, memory_order __m = memory_order_seq_cst) noexcept
  {
    return _CUDA_VSTD::__atomic_fetch_min_dispatch(&this->__a, __op, __m, _CUDA_VSTD::__scope_to_tag<_Sco>{});
  }
  _LIBCUDACXX_HIDE_FROM_ABI _Tp fetch_min(const _Tp& __op, memory_order __m = memory_order_seq_cst) volatile noexcept
  {
    return _CUDA_VSTD::__atomic_fetch_min_dispatch(&this->__a, __op, __m, _CUDA_VSTD::__scope_to_tag<_Sco>{});
  }
};

// atomic_ref<T>

template <class _Tp, thread_scope _Sco = thread_scope::thread_scope_system>
struct atomic_ref : public _CUDA_VSTD::__atomic_ref_impl<_Tp, _Sco>
{
  using value_type = _Tp;

  static constexpr size_t required_alignment = sizeof(_Tp);

  static constexpr bool is_always_lock_free = sizeof(_Tp) <= 8;

  _LIBCUDACXX_HIDE_FROM_ABI explicit constexpr atomic_ref(_Tp& __ref)
      : _CUDA_VSTD::__atomic_ref_impl<_Tp, _Sco>(__ref)
  {}

  _LIBCUDACXX_HIDE_FROM_ABI _Tp operator=(_Tp __v) const noexcept
  {
    this->store(__v);
    return __v;
  }

  _CCCL_HIDE_FROM_ABI atomic_ref(const atomic_ref&) noexcept = default;
  atomic_ref& operator=(const atomic_ref&)                   = delete;
  atomic_ref& operator=(const atomic_ref&) const             = delete;

  _LIBCUDACXX_HIDE_FROM_ABI _Tp fetch_max(const _Tp& __op, memory_order __m = memory_order_seq_cst) const noexcept
  {
    return _CUDA_VSTD::__atomic_fetch_max_dispatch(&this->__a, __op, __m, _CUDA_VSTD::__scope_to_tag<_Sco>{});
  }

  _LIBCUDACXX_HIDE_FROM_ABI _Tp fetch_min(const _Tp& __op, memory_order __m = memory_order_seq_cst) const noexcept
  {
    return _CUDA_VSTD::__atomic_fetch_min_dispatch(&this->__a, __op, __m, _CUDA_VSTD::__scope_to_tag<_Sco>{});
  }
};

inline _CCCL_HOST_DEVICE void
atomic_thread_fence(memory_order __m, thread_scope _Scope = thread_scope::thread_scope_system)
{
  NV_DISPATCH_TARGET(
    NV_IS_DEVICE_LIBHIPCXX,
    (switch (_Scope) {
      case thread_scope::thread_scope_system:
        _CUDA_VSTD::__atomic_thread_fence_cuda((int) __m, __thread_scope_system_tag{});
        break;
      case thread_scope::thread_scope_device:
        _CUDA_VSTD::__atomic_thread_fence_cuda((int) __m, __thread_scope_device_tag{});
        break;
      case thread_scope::thread_scope_block:
        _CUDA_VSTD::__atomic_thread_fence_cuda((int) __m, __thread_scope_block_tag{});
        break;
      // Atomics scoped to themselves do not require fencing
      case thread_scope::thread_scope_thread:
        break;
    }),
    NV_IS_HOST_LIBHIPCXX,
    ((void) _Scope; _CUDA_VSTD::atomic_thread_fence(__m);))
}

inline _CCCL_HOST_DEVICE void atomic_signal_fence(memory_order __m)
{
  _CUDA_VSTD::atomic_signal_fence(__m);
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___ATOMIC_ATOMIC_H

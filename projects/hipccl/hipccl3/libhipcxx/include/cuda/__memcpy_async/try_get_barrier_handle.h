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

#ifndef _CUDA___BARRIER_TRY_GET_BARRIER_HANDLE_H
#define _CUDA___BARRIER_TRY_GET_BARRIER_HANDLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__barrier/barrier_block_scope.h>
#include <cuda/__barrier/barrier_native_handle.h>
#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/__barrier/barrier.h>
#include <cuda/std/__barrier/empty_completion.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstdint>

#include <nv/target>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! @brief __try_get_barrier_handle returns barrier handle of block-scoped barriers and a nullptr otherwise.
template <thread_scope _Sco, typename _CompF>
_LIBCUDACXX_HIDE_FROM_ABI _CUDA_VSTD::uint64_t* __try_get_barrier_handle(barrier<_Sco, _CompF>& __barrier)
{
  return nullptr;
}

template <>
_LIBCUDACXX_HIDE_FROM_ABI _CUDA_VSTD::uint64_t*
__try_get_barrier_handle<::cuda::thread_scope_block, _CUDA_VSTD::__empty_completion>(
  barrier<thread_scope_block>& __barrier)
{
  (void) __barrier;
  NV_DISPATCH_TARGET(
    NV_IS_DEVICE_LIBHIPCXX, (return ::cuda::device::barrier_native_handle(__barrier);), NV_ANY_TARGET_LIBHIPCXX, (return nullptr;));
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___BARRIER_TRY_GET_BARRIER_HANDLE_H

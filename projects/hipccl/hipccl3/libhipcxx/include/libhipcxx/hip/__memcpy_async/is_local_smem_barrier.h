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

#ifndef _CUDA___BARRIER_IS_LOCAL_SMEM_BARRIER_H
#define _CUDA___BARRIER_IS_LOCAL_SMEM_BARRIER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__barrier/barrier.h>
#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/__barrier/empty_completion.h>
#include <cuda/std/__type_traits/is_same.h>

#include <nv/target>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! @brief __is_local_smem_barrier returns true if barrier is (1) block-scoped and (2) located in shared memory.
template <thread_scope _Sco,
          typename _CompF,
          bool _Is_mbarrier = (_Sco == thread_scope_block)
                           && _CCCL_TRAIT(_CUDA_VSTD::is_same, _CompF, _CUDA_VSTD::__empty_completion)>
_LIBCUDACXX_HIDE_FROM_ABI bool __is_local_smem_barrier(barrier<_Sco, _CompF>& __barrier)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE_LIBHIPCXX, (return _Is_mbarrier && __isShared(&__barrier);), (return false;));
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___BARRIER_IS_LOCAL_SMEM_BARRIER_H

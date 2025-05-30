// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

#ifndef __GPU___MUTEX_LOCK_GUARD_H__
#define __GPU___MUTEX_LOCK_GUARD_H__

#include "gpu/__config"

#include <mutex>

namespace gpu {

//====================================================================================================================//
//      Adapted from libc++ std::lock_guard
//====================================================================================================================//

template <class _Mutex>
class _LIBGPU_TEMPLATE_VIS _LIBGPU_THREAD_SAFETY_ANNOTATION(scoped_lockable) lock_guard {
public:
  typedef _Mutex mutex_type;

private:
  mutex_type& __m_;

public:
  _LIBGPU_NODISCARD_EXT __device__ _LIBGPU_HIDE_FROM_ABI explicit lock_guard(mutex_type& __m) _LIBGPU_THREAD_SAFETY_ANNOTATION(acquire_capability(__m))
      : __m_(__m) {
    __m_.lock();
  }

  _LIBGPU_NODISCARD_EXT __device__ _LIBGPU_HIDE_FROM_ABI lock_guard(mutex_type& __m, std::adopt_lock_t)
      _LIBGPU_THREAD_SAFETY_ANNOTATION(requires_capability(__m))
      : __m_(__m) {}
  __device__ _LIBGPU_HIDE_FROM_ABI ~lock_guard() _LIBGPU_THREAD_SAFETY_ANNOTATION(release_capability()) { __m_.unlock(); }

private:
  __device__ lock_guard(lock_guard const&)            = delete;
  __device__ lock_guard& operator=(lock_guard const&) = delete;
};
_LIBGPU_CTAD_SUPPORTED_FOR_TYPE(lock_guard);

} // namespace gpu

#endif // __GPU___MUTEX_LOCK_GUARD_H__

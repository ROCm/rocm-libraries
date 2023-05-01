//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___MUTEX_LOCK_GUARD_H__
#define __GPU___MUTEX_LOCK_GUARD_H__

#include "gpu/__config"

#include <mutex>

namespace gpu {

//====================================================================================================================//
//      Adapted from libc++ std::lock_guard
//====================================================================================================================//

template <class _Mutex>
class _LIBGPU_TEMPLATE_VIS lock_guard {
public:
  typedef _Mutex mutex_type;

private:
  mutex_type& __m_;

public:
  _LIBGPU_NODISCARD_EXT __device__ _LIBGPU_HIDE_FROM_ABI explicit lock_guard(mutex_type& __m)
      : __m_(__m) {
    __m_.lock();
  }

  _LIBGPU_NODISCARD_EXT __device__ _LIBGPU_HIDE_FROM_ABI lock_guard(mutex_type& __m, std::adopt_lock_t)
      : __m_(__m) {}
  __device__ _LIBGPU_HIDE_FROM_ABI ~lock_guard() { __m_.unlock(); }

private:
  __device__ lock_guard(lock_guard const&)            = delete;
  __device__ lock_guard& operator=(lock_guard const&) = delete;
};
_LIBGPU_CTAD_SUPPORTED_FOR_TYPE(lock_guard);

} // namespace gpu

#endif // __GPU___MUTEX_LOCK_GUARD_H__

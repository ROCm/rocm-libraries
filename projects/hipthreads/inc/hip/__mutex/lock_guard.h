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

#ifndef __LIBHIPTHREADS___MUTEX_LOCK_GUARD_H__
#define __LIBHIPTHREADS___MUTEX_LOCK_GUARD_H__

#include "hip/thread_config"

#include <mutex>

/**
 * @file
 * @brief RAII wrapper that acquires a mutex for a scope and releases on destruction.
 * @ingroup mutex
 *
 * Device-side analogue of std::lock_guard. Ensures a BasicLockable mutex is
 * locked for the lifetime of the guard object, providing exception / early
 * return safety.
 *
 * Differences vs std::lock_guard:
 * - Annotated __device__ (no host build path yet).
 * - Same interface otherwise (default locking ctor + adopt_lock tag).
 *
 * Pitfall:
 *   Avoid constructing an unnamed temporary (e.g. lock_guard<spin_mutex>{m};)
 *   because it is destroyed immediately and provides no protection.
 */

namespace cuda {

//====================================================================================================================//
//      Adapted from libc++ ::std::lock_guard
//====================================================================================================================//

 /**
 * @tparam _Mutex Mutex type meeting BasicLockable (lock(), unlock()).
 * @brief Scoped non-copyable guard that owns a mutex for its lifetime.
 * @ingroup mutex
 * 
 * Guarantees:
 * - Acquires the mutex in the locking constructor.
 * - Releases the mutex in the destructor.
 * - Non-copyable and non-assignable to prevent multiple owners.
 */
template <class _Mutex>
class _LIBHIPTHREADS_TEMPLATE_VIS _LIBHIPTHREADS_THREAD_SAFETY_ANNOTATION(scoped_lockable) lock_guard {
public:
  /// Alias for the underlying mutex type.
  typedef _Mutex mutex_type;

private:
  mutex_type& __m_;

public:
  /**
   * @brief Locks the given mutex.
   * @param __m Mutex to lock (must not already be locked by this wthread).
   */
  _LIBHIPTHREADS_NODISCARD_EXT __device__ _LIBHIPTHREADS_HIDE_FROM_ABI explicit lock_guard(mutex_type& __m) _LIBHIPTHREADS_THREAD_SAFETY_ANNOTATION(acquire_capability(__m))
      : __m_(__m) {
    __m_.lock();
  }

  /**
   * @brief Adopts ownership of an already-locked mutex.
   * @param __m Mutex already locked by the calling wthread.
   * @param (adopt_lock) Tag indicating adoption (no lock attempt made).
   * @pre The calling wthread holds the mutex.
   */
  _LIBHIPTHREADS_NODISCARD_EXT __device__ _LIBHIPTHREADS_HIDE_FROM_ABI lock_guard(mutex_type& __m, ::std::adopt_lock_t)
      _LIBHIPTHREADS_THREAD_SAFETY_ANNOTATION(requires_capability(__m))
      : __m_(__m) {}

  /**
   * @brief Unlocks the mutex.
   */
  __device__ _LIBHIPTHREADS_HIDE_FROM_ABI ~lock_guard() _LIBHIPTHREADS_THREAD_SAFETY_ANNOTATION(release_capability()) { __m_.unlock(); }

private:
  __device__ lock_guard(lock_guard const&)            = delete;
  __device__ lock_guard& operator=(lock_guard const&) = delete;
};
_LIBHIPTHREADS_CTAD_SUPPORTED_FOR_TYPE(lock_guard);

} // namespace cuda

#endif // __LIBHIPTHREADS___MUTEX_LOCK_GUARD_H__

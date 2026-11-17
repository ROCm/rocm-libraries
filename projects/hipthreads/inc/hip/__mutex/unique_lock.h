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

#ifndef __LIBHIPTHREADS___MUTEX_UNIQUE_LOCK_H__
#define __LIBHIPTHREADS___MUTEX_UNIQUE_LOCK_H__

#include "hip/thread_config"

#include <mutex>

#include "hip/std/__memory/addressof.h"
#include "hip/std/__utility/swap.h"

/**
 * @file
 * @brief Flexible RAII mutex wrapper supporting deferred / try / adopt locking and move transfer.
 * @ingroup mutex
 *
 * Device-side analogue of std::unique_lock. Compared to lock_guard this adds:
 * - Optional acquisition strategies (immediate lock, defer, try, adopt).
 * - Move semantics (transfer ownership between unique_lock objects).
 * - Explicit lock()/unlock()/try_lock() member functions for use with
 *   condition variables and manual lock management.
 *
 * Planned (not yet enabled): timed locking (try_lock_for / try_lock_until).
 *
 * Warning:
 *   As with any RAII lock, do not create an unnamed temporary
 *   (e.g. unique_lock<spin_mutex>{m};) — it unlocks immediately.
 */

namespace cuda {

//====================================================================================================================//
//      Adapted from libc++ ::std::unique_lock
//====================================================================================================================//

/**
 * @tparam _Mutex Mutex type meeting BasicLockable (lock(), unlock()).
 * @brief Movable (not copyable) lock owning at most one mutex.
 * @ingroup mutex
 *
 * Features:
 * - Multiple constructor tags: (default), defer_lock, try_to_lock, adopt_lock.
 * - Manual lock()/unlock()/try_lock().
 * - Release without unlocking (release()).
 * - Ownership query (owns_lock() / operator bool()).
 * - Move construct / assign transfers ownership; source is left empty.
 *
 * Invariants:
 * - If owns_lock() is true then mutex() is non-null and locked by this wthread.
 * - Destruction unlocks only if ownership is held.
 *
 * Timed locking hooks are commented out (TODO) — will forward to underlying mutex
 * once TimedLockable support is implemented.
 */
template <class _Mutex>
class _LIBHIPTHREADS_TEMPLATE_VIS unique_lock {
public:
  /// Underlying mutex type.
  typedef _Mutex mutex_type;

private:
  mutex_type* __m_;
  bool __owns_;

public:
  /// Constructs an empty non-owning lock (mutex() == nullptr).
  __device__ _LIBHIPTHREADS_HIDE_FROM_ABI unique_lock() _NOEXCEPT : __m_(nullptr), __owns_(false) {}
  
  /**
   * @brief Locks the supplied mutex immediately.
   * @param __m Target mutex (must not already be locked by this wthread).
   * @post owns_lock() == true
   */
  __device__ _LIBHIPTHREADS_HIDE_FROM_ABI explicit unique_lock(mutex_type& __m) : __m_(hip::std::addressof(__m)), __owns_(true) {
    __m_->lock();
  }

  /**
   * @brief Associates with a mutex without locking it yet.
   * @param __m Target mutex.
   * @param (defer_lock) Tag indicating deferred acquisition.
   * @post owns_lock() == false
   */
  __device__ _LIBHIPTHREADS_HIDE_FROM_ABI unique_lock(mutex_type& __m, ::std::defer_lock_t) _NOEXCEPT
      : __m_(hip::std::addressof(__m)),
        __owns_(false) {}

  /**
   * @brief Attempts to lock without blocking.
   * @param __m Target mutex.
   * @param (try_to_lock) Tag requesting a non-blocking attempt.
   * @post owns_lock() reflects success.
   */
  __device__ _LIBHIPTHREADS_HIDE_FROM_ABI unique_lock(mutex_type& __m, ::std::try_to_lock_t)
      : __m_(hip::std::addressof(__m)), __owns_(__m.try_lock()) {}

  /**
   * @brief Assumes caller already holds the mutex.
   * @param __m Target mutex already locked by this wthread.
   * @param (adopt_lock) Tag asserting prior lock ownership.
   * @pre Current wthread owns __m.
   */
  __device__ _LIBHIPTHREADS_HIDE_FROM_ABI unique_lock(mutex_type& __m, ::std::adopt_lock_t) : __m_(hip::std::addressof(__m)), __owns_(true) {}

  // TODO: Uncomment this once we implement chrono
  // template <class _Clock, class _Duration>
  // __device__ _LIBHIPTHREADS_HIDE_FROM_ABI unique_lock(mutex_type& __m, const chrono::time_point<_Clock, _Duration>& __t)
  //     : __m_(hip::addressof(__m)), __owns_(__m.try_lock_until(__t)) {}

  // TODO: Uncomment this once we implement chrono
  // template <class _Rep, class _Period>
  // __device__ _LIBHIPTHREADS_HIDE_FROM_ABI unique_lock(mutex_type& __m, const chrono::duration<_Rep, _Period>& __d)
  //     : __m_(hip::addressof(__m)), __owns_(__m.try_lock_for(__d)) {}

  /// Destructor unlocks if ownership is held.
  __device__ _LIBHIPTHREADS_HIDE_FROM_ABI ~unique_lock() {
    if (__owns_)
      __m_->unlock();
  }

  /// \name Deleted copy operations
  /// Instances are movable but not copyable.
  ///@{
  __device__ unique_lock(unique_lock const&)            = delete;
  __device__ unique_lock& operator=(unique_lock const&) = delete;
  ///@}

  /**
   * @brief Move constructs, transferring ownership.
   * @param __u Source; becomes empty afterward.
   */
  __device__ _LIBHIPTHREADS_HIDE_FROM_ABI unique_lock(unique_lock&& __u) _NOEXCEPT : __m_(__u.__m_), __owns_(__u.__owns_) {
    __u.__m_    = nullptr;
    __u.__owns_ = false;
  }

  /**
   * @brief Move assigns, releasing any owned mutex then taking __u's.
   * @param __u Source; becomes empty.
   * @return *this
   */
  __device__ _LIBHIPTHREADS_HIDE_FROM_ABI unique_lock& operator=(unique_lock&& __u) _NOEXCEPT {
    if (__owns_)
      __m_->unlock();

    __m_        = __u.__m_;
    __owns_     = __u.__owns_;
    __u.__m_    = nullptr;
    __u.__owns_ = false;
    return *this;
  }

  /**
   * @brief Locks the associated mutex (must be associated & not owned).
   * @throws (asserts in device build on misuse).
   * @post owns_lock() == true
   */
  __device__ void lock();

  /**
   * @brief Attempts to lock without blocking.
   * @return true if the lock was obtained.
   * @post owns_lock() reflects result.
   */
  __device__ bool try_lock();

  // TODO: Uncomment this once we implement chrono
  // template <class _Rep, class _Period>
  // __device__ bool try_lock_for(const chrono::duration<_Rep, _Period>& __d);

  // TODO: Uncomment this once we implement chrono
  // template <class _Clock, class _Duration>
  // __device__ bool try_lock_until(const chrono::time_point<_Clock, _Duration>& __t);

  /**
   * @brief Unlocks the mutex.
   * @pre owns_lock() == true
   * @post owns_lock() == false
   */
  __device__ void unlock();

  /**
   * @brief Exchanges state with another unique_lock.
   * @param __u Other lock.
   * @post Ownership flags and mutex pointers swapped.
   */
  __device__ _LIBHIPTHREADS_HIDE_FROM_ABI void swap(unique_lock& __u) _NOEXCEPT {
    hip::std::swap(__m_, __u.__m_);
    hip::std::swap(__owns_, __u.__owns_);
  }

  /**
   * @brief Releases association without unlocking.
   * @return Previously associated mutex pointer (may be null).
   * @post mutex()==nullptr && owns_lock()==false
   */
  __device__ _LIBHIPTHREADS_HIDE_FROM_ABI mutex_type* release() _NOEXCEPT {
    mutex_type* __m = __m_;
    __m_            = nullptr;
    __owns_         = false;
    return __m;
  }

  /// True if this object currently owns the mutex.
  __device__ _LIBHIPTHREADS_HIDE_FROM_ABI bool owns_lock() const _NOEXCEPT { return __owns_; }
  /// Same as owns_lock().
  __device__ _LIBHIPTHREADS_HIDE_FROM_ABI explicit operator bool() const _NOEXCEPT { return __owns_; }
  /// Pointer to the associated mutex (may be null if default-constructed or released).
  __device__ _LIBHIPTHREADS_HIDE_FROM_ABI mutex_type* mutex() const _NOEXCEPT { return __m_; }
};
_LIBHIPTHREADS_CTAD_SUPPORTED_FOR_TYPE(unique_lock);

template <class _Mutex>
__device__ void unique_lock<_Mutex>::lock() {
  assert(__m_ != nullptr && "unique_lock::lock: references null mutex");
  assert(!__owns_ && "unique_lock::lock: already locked");
  __m_->lock();
  __owns_ = true;
}

template <class _Mutex>
__device__ bool unique_lock<_Mutex>::try_lock() {
  assert(__m_ != nullptr && "unique_lock::try_lock: references null mutex");
  assert(!__owns_ && "unique_lock::try_lock: already locked");
  __owns_ = __m_->try_lock();
  return __owns_;
}

// TODO: Uncomment this once we implement chrono
// template <class _Mutex>
// template <class _Rep, class _Period>
// __device__ bool unique_lock<_Mutex>::try_lock_for(const chrono::duration<_Rep, _Period>& __d) {
//   assert(__m_ != nullptr && "unique_lock::try_lock_for: references null mutex");
//   assert(!__owns_ && "unique_lock::try_lock_for: already locked");
//   __owns_ = __m_->try_lock_for(__d);
//   return __owns_;
// }

// TODO: Uncomment this once we implement chrono
// template <class _Mutex>
// template <class _Clock, class _Duration>
// __device__ bool unique_lock<_Mutex>::try_lock_until(const chrono::time_point<_Clock, _Duration>& __t) {
//   assert(__m_ != nullptr && "unique_lock::try_lock_until: references null mutex");
//   assert(!__owns_ && "unique_lock::try_lock_until: already locked");
//   __owns_ = __m_->try_lock_until(__t);
//   return __owns_;
// }

template <class _Mutex>
__device__ void unique_lock<_Mutex>::unlock() {
  assert(__owns_ && "unique_lock::unlock: not locked");
  __m_->unlock();
  __owns_ = false;
}

template <class _Mutex>
__device__ inline _LIBHIPTHREADS_HIDE_FROM_ABI void swap(unique_lock<_Mutex>& __x, unique_lock<_Mutex>& __y) _NOEXCEPT {
  __x.swap(__y);
}

} // namespace cuda

#endif // __LIBHIPTHREADS___MUTEX_UNIQUE_LOCK_H__

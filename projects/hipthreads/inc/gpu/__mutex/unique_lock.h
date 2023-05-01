//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___MUTEX_UNIQUE_LOCK_H__
#define __GPU___MUTEX_UNIQUE_LOCK_H__

#include "gpu/__config"

#include <mutex>

#include "gpu/__memory/addressof.h"
#include "gpu/__utility/swap.h"

namespace gpu {

//====================================================================================================================//
//      Adapted from libc++ std::unique_lock
//====================================================================================================================//

template <class _Mutex>
class _LIBGPU_TEMPLATE_VIS unique_lock {
public:
  typedef _Mutex mutex_type;

private:
  mutex_type* __m_;
  bool __owns_;

public:
  __device__ _LIBGPU_HIDE_FROM_ABI unique_lock() _NOEXCEPT : __m_(nullptr), __owns_(false) {}
  __device__ _LIBGPU_HIDE_FROM_ABI explicit unique_lock(mutex_type& __m) : __m_(gpu::addressof(__m)), __owns_(true) {
    __m_->lock();
  }

  __device__ _LIBGPU_HIDE_FROM_ABI unique_lock(mutex_type& __m, std::defer_lock_t) _NOEXCEPT
      : __m_(gpu::addressof(__m)),
        __owns_(false) {}

  __device__ _LIBGPU_HIDE_FROM_ABI unique_lock(mutex_type& __m, std::try_to_lock_t)
      : __m_(gpu::addressof(__m)), __owns_(__m.try_lock()) {}

  __device__ _LIBGPU_HIDE_FROM_ABI unique_lock(mutex_type& __m, std::adopt_lock_t) : __m_(gpu::addressof(__m)), __owns_(true) {}

  // TODO: Uncomment this once we implement chrono
  // template <class _Clock, class _Duration>
  // __device__ _LIBGPU_HIDE_FROM_ABI unique_lock(mutex_type& __m, const chrono::time_point<_Clock, _Duration>& __t)
  //     : __m_(gpu::addressof(__m)), __owns_(__m.try_lock_until(__t)) {}

  // TODO: Uncomment this once we implement chrono
  // template <class _Rep, class _Period>
  // __device__ _LIBGPU_HIDE_FROM_ABI unique_lock(mutex_type& __m, const chrono::duration<_Rep, _Period>& __d)
  //     : __m_(gpu::addressof(__m)), __owns_(__m.try_lock_for(__d)) {}

  __device__ _LIBGPU_HIDE_FROM_ABI ~unique_lock() {
    if (__owns_)
      __m_->unlock();
  }

  __device__ unique_lock(unique_lock const&)            = delete;
  __device__ unique_lock& operator=(unique_lock const&) = delete;

  __device__ _LIBGPU_HIDE_FROM_ABI unique_lock(unique_lock&& __u) _NOEXCEPT : __m_(__u.__m_), __owns_(__u.__owns_) {
    __u.__m_    = nullptr;
    __u.__owns_ = false;
  }

  __device__ _LIBGPU_HIDE_FROM_ABI unique_lock& operator=(unique_lock&& __u) _NOEXCEPT {
    if (__owns_)
      __m_->unlock();

    __m_        = __u.__m_;
    __owns_     = __u.__owns_;
    __u.__m_    = nullptr;
    __u.__owns_ = false;
    return *this;
  }

  __device__ void lock();
  __device__ bool try_lock();

  // TODO: Uncomment this once we implement chrono
  // template <class _Rep, class _Period>
  // __device__ bool try_lock_for(const chrono::duration<_Rep, _Period>& __d);

  // TODO: Uncomment this once we implement chrono
  // template <class _Clock, class _Duration>
  // __device__ bool try_lock_until(const chrono::time_point<_Clock, _Duration>& __t);

  __device__ void unlock();

  __device__ _LIBGPU_HIDE_FROM_ABI void swap(unique_lock& __u) _NOEXCEPT {
    gpu::swap(__m_, __u.__m_);
    gpu::swap(__owns_, __u.__owns_);
  }

  __device__ _LIBGPU_HIDE_FROM_ABI mutex_type* release() _NOEXCEPT {
    mutex_type* __m = __m_;
    __m_            = nullptr;
    __owns_         = false;
    return __m;
  }

  __device__ _LIBGPU_HIDE_FROM_ABI bool owns_lock() const _NOEXCEPT { return __owns_; }
  __device__ _LIBGPU_HIDE_FROM_ABI explicit operator bool() const _NOEXCEPT { return __owns_; }
  __device__ _LIBGPU_HIDE_FROM_ABI mutex_type* mutex() const _NOEXCEPT { return __m_; }
};
_LIBGPU_CTAD_SUPPORTED_FOR_TYPE(unique_lock);

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
__device__ inline _LIBGPU_HIDE_FROM_ABI void swap(unique_lock<_Mutex>& __x, unique_lock<_Mutex>& __y) _NOEXCEPT {
  __x.swap(__y);
}

} // namespace gpu

#endif // __GPU___MUTEX_UNIQUE_LOCK_H__

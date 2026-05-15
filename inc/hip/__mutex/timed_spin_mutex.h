// -*- C++ -*-

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

#ifndef __LIBHIPTHREADS___MUTEX_TIMED_SPIN_MUTEX_H__
#define __LIBHIPTHREADS___MUTEX_TIMED_SPIN_MUTEX_H__

#include "hip/thread_config"

#include "hip/__condition_variable/spin_condition_variable.h"
#include "hip/__mutex/spin_mutex.h"
#include "hip/__mutex/unique_lock.h"

/**
 * @file
 * @brief Timed spin mutex: spin_mutex extended with try_lock_for / try_lock_until.
 * @ingroup mutex
 */

namespace cuda {

//====================================================================================================================//
//      Adapted from libc++ ::std::timed_mutex
//====================================================================================================================//

/**
 * @class timed_spin_mutex
 * @brief Busy-wait (spin) mutex with timed lock acquisition.
 * @ingroup mutex
 *
 * Characteristics:
 * - Exclusive, non-recursive ownership.
 * - lock() spins until acquired (via internal spin_mutex + spin_condition_variable).
 * - try_lock() single non-blocking attempt.
 * - try_lock_for() / try_lock_until() attempt acquisition until a deadline.
 * - unlock() releases ownership and notifies one waiter.
 *
 * Meets the TimedLockable requirements.
 * Not copyable or movable.
 */
class _LIBHIPTHREADS_TYPE_VIS timed_spin_mutex {
    spin_mutex __m_;
    spin_condition_variable __cv_;
    bool __locked_;

public:
    /// Constructs an unlocked timed_spin_mutex.
    __device__ _LIBHIPTHREADS_HIDE_FROM_ABI timed_spin_mutex() : __locked_(false) {}

    /// \name Deleted copy / move operations
    ///@{
    __device__ timed_spin_mutex(const timed_spin_mutex &) = delete;
    __device__ timed_spin_mutex &operator=(const timed_spin_mutex &) = delete;
    ///@}

    __device__ _LIBHIPTHREADS_HIDE_FROM_ABI ~timed_spin_mutex() { lock_guard<spin_mutex> __lk(__m_); }

    /// @brief Acquires the mutex, spinning until available.
    __device__ _LIBHIPTHREADS_HIDE_FROM_ABI void lock() {
        unique_lock<spin_mutex> __lk(__m_);
        while (__locked_)
            __cv_.wait(__lk);
        __locked_ = true;
    }

    /// @brief Attempts a single non-blocking acquisition.
    /// @return true if ownership obtained, false if already locked.
    __device__ _LIBHIPTHREADS_HIDE_FROM_ABI bool try_lock() _NOEXCEPT {
        unique_lock<spin_mutex> __lk(__m_, ::std::try_to_lock);
        if (__lk.owns_lock() && !__locked_) {
            __locked_ = true;
            return true;
        }
        return false;
    }

    /// @brief Attempts acquisition until abs_time is reached.
    /// @return true if ownership obtained before deadline, false on timeout.
    template <class _Clock, class _Duration>
    __device__ _LIBHIPTHREADS_HIDE_FROM_ABI bool
    try_lock_until(const hip::std::chrono::time_point<_Clock, _Duration> &__t) {
        unique_lock<spin_mutex> __lk(__m_);
        bool __no_timeout = _Clock::now() < __t;
        while (__no_timeout && __locked_) {
            __no_timeout = __cv_.wait_until(__lk, __t) == ::std::cv_status::no_timeout;
        }
        if (!__locked_) {
            __locked_ = true;
            return true;
        }
        return false;
    }

    /// @brief Attempts acquisition for a relative duration.
    /// @return true if ownership obtained within duration, false on timeout.
    template <class _Rep, class _Period>
    __device__ _LIBHIPTHREADS_HIDE_FROM_ABI bool try_lock_for(const hip::std::chrono::duration<_Rep, _Period> &__d) {
        return try_lock_until(hip::std::chrono::system_clock::now() + __d);
    }

    /// @brief Releases ownership and notifies one waiter.
    __device__ _LIBHIPTHREADS_HIDE_FROM_ABI void unlock() _NOEXCEPT {
        lock_guard<spin_mutex> __lk(__m_);
        __locked_ = false;
        __cv_.notify_one();
    }
};

} // namespace cuda

#endif // __LIBHIPTHREADS___MUTEX_TIMED_SPIN_MUTEX_H__

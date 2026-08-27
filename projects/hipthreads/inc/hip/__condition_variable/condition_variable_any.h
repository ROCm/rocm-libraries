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

#ifndef __LIBHIPTHREADS___CONDITION_VARIABLE_CONDITION_VARIABLE_ANY_H__
#define __LIBHIPTHREADS___CONDITION_VARIABLE_CONDITION_VARIABLE_ANY_H__

/**
 * @file
 * @brief Generic condition variable usable with any lock type meeting BasicLockable.
 * @ingroup condition_variable
 *
 * Device-side analogue of std::condition_variable_any. Provides wait()/notify
 * coordination where wait() releases the supplied lock and reacquires it before
 * returning.
 *
 * Characteristics / differences vs the standard host version:
 * - Implemented with spinning (no true sleep); suitable only when expected
 *   waits are short.
 * - Time-based waits (wait_for / wait_until) are not yet enabled (chrono TODOs).
 * - All current members are device-qualified (__device__).
 * - Spurious wakeups can occur: always use the predicate overload when waiting
 *   for a condition.
 */

#include "hip/thread_config"

#include "hip/hip_runtime.h" // Atomics aren't part of hip_runtime_api.h

namespace cuda {

/**
 * @brief Condition variable that works with any BasicLockable lock.
 * @ingroup condition_variable
 *
 * Maintains separate counters for waiters and notifications. Each wait()
 * obtains an arrival index; notify operations advance a notification counter.
 *
 * @note Not copyable or movable.
 * @warning Implementation spins while waiting; avoid long waits to prevent
 *          excessive resource usage.
 */
class _LIBHIPTHREADS_TYPE_VIS condition_variable_any {
    uint64_t wait_counter = 0;
    uint64_t notify_counter = 0;

  public:
    /// Constructs an empty condition variable (no waiters).
    __device__ _LIBHIPTHREADS_HIDE_FROM_ABI _LIBHIPTHREADS_CONSTEXPR condition_variable_any() _NOEXCEPT = default;

    /// \name Deleted copy / move operations
    /// Condition variable objects are neither copyable nor movable.
    ///@{
    __device__ condition_variable_any(const condition_variable_any &) = delete;
    __device__ condition_variable_any &operator=(const condition_variable_any &) = delete;
    ///@}

    /**
     * @brief Wakes at most one waiting wthread/work item (if any).
     *
     * If no waiter is currently blocked, the call is a no-op (no "stored"
     * wake token).
     */
    __device__ _LIBHIPTHREADS_HIDE_FROM_ABI void notify_one() _NOEXCEPT;

    /**
     * @brief Wakes all current waiters.
     *
     * Advances the notification counter to match the waiter counter so every
     * waiter observing the value will proceed.
     */
    __device__ _LIBHIPTHREADS_HIDE_FROM_ABI void notify_all() _NOEXCEPT;

    /**
     * @brief Blocks (spins) until notified.
     *
     * Atomically:
     * 1. Records arrival index.
     * 2. Releases the supplied lock (lock.unlock()).
     * 3. Spins until its index is < current notify counter.
     * 4. Reacquires the lock before returning.
     *
     * @tparam _Lock Lock type supporting unlock() / lock().
     * @param __lock Acquired lock protecting the predicate.
     * @warning Spurious wakeups are possible—prefer the predicate form.
     */
    template <class _Lock>
    __device__ _LIBHIPTHREADS_METHOD_TEMPLATE_IMPLICIT_INSTANTIATION_VIS void wait(_Lock &__lock);

    /**
     * @brief Waits until predicate returns true.
     *
     * Repeatedly invokes the simple wait() and rechecks __pred() under the
     * lock. Returns only when __pred() evaluates to true.
     *
     * @tparam _Lock Lock type.
     * @tparam _Predicate Callable returning bool, evaluated with the lock held.
     * @param __lock Acquired lock.
     * @param __pred Predicate defining the wake condition.
     */
    template <class _Lock, class _Predicate>
    __device__ _LIBHIPTHREADS_HIDE_FROM_ABI void wait(_Lock &__lock, _Predicate __pred);

    // TODO: uncomment these once we implement chrono
    // template <class _Lock, class _Clock, class _Duration>
    //     _LIBHIPTHREADS_METHOD_TEMPLATE_IMPLICIT_INSTANTIATION_VIS
    //     ::std::cv_status
    //     wait_until(_Lock& __lock,
    //                const chrono::time_point<_Clock, _Duration>& __t);

    // template <class _Lock, class _Clock, class _Duration, class _Predicate>
    //     bool
    //     __device__ _LIBHIPTHREADS_HIDE_FROM_ABI
    //     wait_until(_Lock& __lock,
    //                const chrono::time_point<_Clock, _Duration>& __t,
    //                _Predicate __pred);

    // template <class _Lock, class _Rep, class _Period>
    //     ::std::cv_status
    //     __device__ _LIBHIPTHREADS_HIDE_FROM_ABI
    //     wait_for(_Lock& __lock,
    //              const chrono::duration<_Rep, _Period>& __d);

    // template <class _Lock, class _Rep, class _Period, class _Predicate>
    //     bool
    //     __device__ _LIBHIPTHREADS_HIDE_FROM_ABI
    //     wait_for(_Lock& __lock,
    //              const chrono::duration<_Rep, _Period>& __d,
    //              _Predicate __pred);
};

__device__ inline void condition_variable_any::notify_one() _NOEXCEPT {
    // If (notify_counter + 1 <= wait_counter), increment notify_counter.
    // If we increment notify_counter when nobody is waiting, then the next person to wait will skip waiting
    uint64_t cached_ntfy_cnt;
    do {
        cached_ntfy_cnt = atomicAdd(&notify_counter, 0);
        if (cached_ntfy_cnt >= atomicAdd(&wait_counter, 0))
            return;
    } while (atomicCAS(&notify_counter, cached_ntfy_cnt, cached_ntfy_cnt + 1) !=
             cached_ntfy_cnt);
}

__device__ inline void condition_variable_any::notify_all() _NOEXCEPT {
    atomicExch(&notify_counter, atomicAdd(&wait_counter, 0));
}

template <class _Lock>
__device__ void condition_variable_any::wait(_Lock &__lock) {
    uint64_t myId = atomicAdd(&wait_counter, 1);
    // It's possible that another thread calls notify here, and then checks the state of __lock before we get a chance
    // to unlock it. Since we're supposed to ATOMICALLY unlock and 'sleep', it technically shouldn't be possible for
    // another thread to 'wake' us before we've released the lock. However, from a user's perspective, this situation is
    // nearly indistinguisable from another, perfectly legal occurence: were already a bit further ahead, in the loop
    // 'sleeping' when the other thread called notify, then woke up and re-acquired the lock before they got the chance
    // to do anything further. The only catch is if unlocking and re-locking a lock has side effects or doesn't return
    // it to an identical state, the user might be able to differentiate between these two situations.

    // This isn't a big deal though. We can't create a perfect replacement for condition variable anyways because we
    // can't actually 'sleep'. If we really wanted to, we could safely use this implementation for
    // spin_condition_variable only, and implement condition_variable_any using that (like libcxx uses
    // ::std::condition_variable), but that would significantly hurt performance.
    __lock.unlock();
    while (myId >= atomicAdd(&notify_counter, 0)) {
        // __threadfence();
        // TODO: should we sleep here?
        // __builtin_amdgcn_s_sleep(8);
    }
    __lock.lock();
}

template <class _Lock, class _Predicate>
__device__ inline void condition_variable_any::wait(_Lock &__lock, _Predicate __pred) {
    while (!__pred())
        wait(__lock);
}

} // namespace cuda

#endif // __LIBHIPTHREADS___CONDITION_VARIABLE_CONDITION_VARIABLE_ANY_H__

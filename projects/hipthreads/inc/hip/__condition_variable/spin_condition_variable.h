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

#ifndef __LIBHIPTHREADS___CONDITION_VARIABLE_SPIN_CONDITION_VARIABLE_H__
#define __LIBHIPTHREADS___CONDITION_VARIABLE_SPIN_CONDITION_VARIABLE_H__

/**
 * @file
 * @brief Specialized spinning condition variable restricted to spin_mutex.
 * @ingroup condition_variable
 *
 * One of two device-side analogues of std::condition_variable (the other one being pseudo_condition_variable).
 *
 * Currently implemented as a thin adapter over condition_variable_any that:
 * - Restricts waits to unique_lock<spin_mutex> (no templates exposed here).
 * - Inlines the fast path so the compiler can optimize for a known lock type.
 * - Retains pure spinning semantics (no blocking / sleep).
 *
 * Limitations:
 * - No timed waits yet.
 * - Busy-waits: prolonged contention wastes GPU execution resources.
 */

#include "hip/thread_config"

#include "hip/hip_runtime_api.h"

#include "hip/__condition_variable/condition_variable_any.h"
#include "hip/__mutex/spin_mutex.h"
#include "hip/__mutex/unique_lock.h"

namespace cuda {

/**
 * @brief Spin-based condition variable bound to spin_mutex.
 * @ingroup condition_variable
 *
 * Privately inherits condition_variable_any to reuse its counter logic,
 * re-exposing notify and tailored wait overloads for unique_lock<spin_mutex>.
 *
 * @note Not copyable.
 * @warning Pure spinning; avoid long waits to reduce resource burn.
 */
class _LIBHIPTHREADS_TYPE_VIS spin_condition_variable : private condition_variable_any {
  public:
    /// Constructs an empty spin condition variable.
    __device__ _LIBHIPTHREADS_HIDE_FROM_ABI _LIBHIPTHREADS_CONSTEXPR spin_condition_variable() _NOEXCEPT = default;

    /// \name Deleted copy / move operations
    /// Instances are neither copyable nor movable.
    ///@{
    __device__ spin_condition_variable(const spin_condition_variable &) = delete;
    __device__ spin_condition_variable &operator=(const spin_condition_variable &) = delete;
    ///@}

    using condition_variable_any::notify_all;
    using condition_variable_any::notify_one;

    /**
     * @brief Waits (spins) until notified.
     *
     * Releases the lock, spins polling the internal counters, then
     * reacquires before returning.
     *
     * @param __lk Acquired unique_lock guarding the predicate.
     */
    __device__ void wait(unique_lock<spin_mutex> &__lk) _NOEXCEPT {
        condition_variable_any::wait(__lk);
    }

    /**
     * @brief Waits until predicate returns true.
     *
     * Repeatedly performs wait() then rechecks __pred() under the lock.
     *
     * @tparam _Predicate Callable returning bool.
     * @param __lk Acquired unique_lock.
     * @param __pred Predicate tested after each wake/spin cycle.
     */
    template <class _Predicate>
    __device__ _LIBHIPTHREADS_METHOD_TEMPLATE_IMPLICIT_INSTANTIATION_VIS void wait(unique_lock<spin_mutex> &__lk, _Predicate __pred) {
        condition_variable_any::wait(__lk, __pred);
    }

    // TODO: Uncomment these once we've implemented chrono
    // template <class _Clock, class _Duration>
    // __device__ _LIBHIPTHREADS_METHOD_TEMPLATE_IMPLICIT_INSTANTIATION_VIS ::std::cv_status
    // wait_until(unique_lock<spin_mutex>& __lk, const chrono::time_point<_Clock, _Duration>& __t);

    // template <class _Clock, class _Duration, class _Predicate>
    // __device__ _LIBHIPTHREADS_METHOD_TEMPLATE_IMPLICIT_INSTANTIATION_VIS bool
    // wait_until(unique_lock<spin_mutex>& __lk, const chrono::time_point<_Clock, _Duration>& __t, _Predicate __pred);

    // template <class _Rep, class _Period>
    // __device__ _LIBHIPTHREADS_METHOD_TEMPLATE_IMPLICIT_INSTANTIATION_VIS ::std::cv_status
    // wait_for(unique_lock<spin_mutex>& __lk, const chrono::duration<_Rep, _Period>& __d);

    // template <class _Rep, class _Period, class _Predicate>
    // __device__ bool _LIBHIPTHREADS_HIDE_FROM_ABI
    // wait_for(unique_lock<spin_mutex>& __lk, const chrono::duration<_Rep, _Period>& __d, _Predicate __pred);
};

} // namespace cuda

#endif // __LIBHIPTHREADS___CONDITION_VARIABLE_SPIN_CONDITION_VARIABLE_H__

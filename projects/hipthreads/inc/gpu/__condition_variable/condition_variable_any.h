// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___CONDITION_VARIABLE_CONDITION_VARIABLE_ANY_H__
#define __GPU___CONDITION_VARIABLE_CONDITION_VARIABLE_ANY_H__

#include "gpu/__config"

#include "hip/hip_runtime.h" // Atomics aren't part of hip_runtime_api.h

namespace gpu {

class _LIBGPU_TYPE_VIS condition_variable_any {
    uint64_t wait_counter = 0;
    uint64_t notify_counter = 0;

  public:
    __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR condition_variable_any() _NOEXCEPT = default;

    __device__ condition_variable_any(const condition_variable_any &) = delete;
    __device__ condition_variable_any &operator=(const condition_variable_any &) = delete;

    __device__ _LIBGPU_HIDE_FROM_ABI void notify_one() _NOEXCEPT;
    __device__ _LIBGPU_HIDE_FROM_ABI void notify_all() _NOEXCEPT;

    template <class _Lock>
    __device__ _LIBGPU_METHOD_TEMPLATE_IMPLICIT_INSTANTIATION_VIS void wait(_Lock &__lock);
    template <class _Lock, class _Predicate>
    __device__ _LIBGPU_HIDE_FROM_ABI void wait(_Lock &__lock, _Predicate __pred);

    // TODO: uncomment these once we implement chrono
    // template <class _Lock, class _Clock, class _Duration>
    //     _LIBGPU_METHOD_TEMPLATE_IMPLICIT_INSTANTIATION_VIS
    //     std::cv_status
    //     wait_until(_Lock& __lock,
    //                const chrono::time_point<_Clock, _Duration>& __t);

    // template <class _Lock, class _Clock, class _Duration, class _Predicate>
    //     bool
    //     __device__ _LIBGPU_HIDE_FROM_ABI
    //     wait_until(_Lock& __lock,
    //                const chrono::time_point<_Clock, _Duration>& __t,
    //                _Predicate __pred);

    // template <class _Lock, class _Rep, class _Period>
    //     std::cv_status
    //     __device__ _LIBGPU_HIDE_FROM_ABI
    //     wait_for(_Lock& __lock,
    //              const chrono::duration<_Rep, _Period>& __d);

    // template <class _Lock, class _Rep, class _Period, class _Predicate>
    //     bool
    //     __device__ _LIBGPU_HIDE_FROM_ABI
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
    // std::condition_variable), but that would significantly hurt performance.
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

} // namespace gpu

#endif // __GPU___CONDITION_VARIABLE_CONDITION_VARIABLE_ANY_H__

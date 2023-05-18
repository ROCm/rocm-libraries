//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___CONDITION_VARIABLE_SPIN_CONDITION_VARIABLE_H__
#define __GPU___CONDITION_VARIABLE_SPIN_CONDITION_VARIABLE_H__

#include "gpu/__config"

#include "hip/hip_runtime_api.h"

#include "gpu/__condition_variable/condition_variable_any.h"
#include "gpu/__mutex/spin_mutex.h"
#include "gpu/__mutex/unique_lock.h"

namespace gpu {

class _LIBGPU_TYPE_VIS spin_condition_variable : private condition_variable_any {
  public:
    __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR spin_condition_variable() _NOEXCEPT = default;

    __device__ spin_condition_variable(const spin_condition_variable &) = delete;
    __device__ spin_condition_variable &operator=(const spin_condition_variable &) = delete;

    using condition_variable_any::notify_all;
    using condition_variable_any::notify_one;

    __device__ void wait(unique_lock<spin_mutex> &__lk) _NOEXCEPT {
        condition_variable_any::wait(__lk);
    }
    template <class _Predicate>
    __device__ _LIBGPU_METHOD_TEMPLATE_IMPLICIT_INSTANTIATION_VIS void wait(unique_lock<spin_mutex> &__lk, _Predicate __pred) {
        condition_variable_any::wait(__lk, __pred);
    }

    // TODO: Uncomment these once we've implemented chrono
    // template <class _Clock, class _Duration>
    // __device__ _LIBGPU_METHOD_TEMPLATE_IMPLICIT_INSTANTIATION_VIS std::cv_status
    // wait_until(unique_lock<spin_mutex>& __lk, const chrono::time_point<_Clock, _Duration>& __t);

    // template <class _Clock, class _Duration, class _Predicate>
    // __device__ _LIBGPU_METHOD_TEMPLATE_IMPLICIT_INSTANTIATION_VIS bool
    // wait_until(unique_lock<spin_mutex>& __lk, const chrono::time_point<_Clock, _Duration>& __t, _Predicate __pred);

    // template <class _Rep, class _Period>
    // __device__ _LIBGPU_METHOD_TEMPLATE_IMPLICIT_INSTANTIATION_VIS std::cv_status
    // wait_for(unique_lock<spin_mutex>& __lk, const chrono::duration<_Rep, _Period>& __d);

    // template <class _Rep, class _Period, class _Predicate>
    // __device__ bool _LIBGPU_HIDE_FROM_ABI
    // wait_for(unique_lock<spin_mutex>& __lk, const chrono::duration<_Rep, _Period>& __d, _Predicate __pred);
};

} // namespace gpu

#endif // __GPU___CONDITION_VARIABLE_SPIN_CONDITION_VARIABLE_H__

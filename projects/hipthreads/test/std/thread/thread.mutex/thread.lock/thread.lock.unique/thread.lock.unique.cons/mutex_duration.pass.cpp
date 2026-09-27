//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -DTEST_NO_HIP_THREAD
// XFAIL: *
// REASON: unique_lock timed constructors not yet implemented in hipThreads

// <mutex>

// template <class Rep, class Period>
// unique_lock::unique_lock(mutex_type& m, const chrono::duration<Rep, Period>& rel_time);

#include <cassert>
#include <hip/std/chrono>
#include <hip/mutex>

#include "force_include_hip.h"

#include "checking_mutex.h"

int main(int, char**) {
#ifdef __HIP_DEVICE_COMPILE__
  checking_mutex mux;
  { // check successful lock
    mux.reject = false;
    hip::unique_lock<checking_mutex> lock(mux, cuda::std::chrono::seconds());
    assert(mux.current_state == checking_mutex::locked_via_try_lock_for);
    assert(lock.owns_lock());
  }
  assert(mux.current_state == checking_mutex::unlocked);

  { // check unsuccessful lock
    mux.reject = true;
    hip::unique_lock<checking_mutex> lock(mux, cuda::std::chrono::seconds());
    assert(mux.current_state == checking_mutex::unlocked);
    assert(mux.last_try == checking_mutex::locked_via_try_lock_for);
    assert(!lock.owns_lock());
  }
  assert(mux.current_state == checking_mutex::unlocked);
#endif

  return 0;
}

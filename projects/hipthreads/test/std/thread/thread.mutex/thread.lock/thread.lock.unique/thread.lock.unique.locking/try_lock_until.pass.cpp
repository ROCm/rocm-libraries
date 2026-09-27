//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -DTEST_NO_HIP_THREAD
// XFAIL: *
// REASON: unique_lock::try_lock_until() not yet implemented in hipThreads

// <mutex>

// template <class Mutex> class unique_lock;

// template <class Clock, class Duration>
//   bool try_lock_until(const chrono::time_point<Clock, Duration>& abs_time);

#include <cassert>
#include <hip/std/chrono>
#include <mutex>
#include <system_error>
#include <hip/mutex>

#include "checking_mutex.h"
#include "test_macros.h"

int main(int, char**) {
  typedef cuda::std::chrono::system_clock Clock;
  checking_mutex mux;

  hip::unique_lock<checking_mutex> lock(mux, ::std::defer_lock_t());

  assert(lock.try_lock_until(Clock::now()));
  assert(mux.current_state == checking_mutex::locked_via_try_lock_until);
  assert(lock.owns_lock());

#ifndef TEST_HAS_NO_EXCEPTIONS
  try {
    mux.last_try = checking_mutex::none;
    (void)lock.try_lock_until(Clock::now());
    assert(false);
  } catch (::std::system_error& e) {
    assert(mux.last_try == checking_mutex::none);
    assert(e.code() == ::std::errc::resource_deadlock_would_occur);
  }
#endif

  lock.unlock();
  mux.reject = true;
  assert(!lock.try_lock_until(Clock::now()));
  assert(mux.last_try == checking_mutex::locked_via_try_lock_until);
  assert(lock.owns_lock() == false);
  lock.release();

#ifndef TEST_HAS_NO_EXCEPTIONS
  try {
    mux.last_try = checking_mutex::none;
    (void)lock.try_lock_until(Clock::now());
    assert(false);
  } catch (::std::system_error& e) {
    assert(mux.last_try == checking_mutex::none);
    assert(e.code() == ::std::errc::operation_not_permitted);
  }
#endif

  return 0;
}

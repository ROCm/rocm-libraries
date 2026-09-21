//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <mutex>

// template <class Mutex> class unique_lock;

// void unlock();

// ADDITIONAL_COMPILE_FLAGS: -DTEST_NO_HIP_THREAD

#include <cassert>
#include <system_error>
#include <hip/mutex>

#include "force_include_hip.h"

#include "test_macros.h"
#include "checking_mutex.h"

int main(int, char**) {
#ifdef __HIP_DEVICE_COMPILE__
  checking_mutex mux;
  hip::unique_lock<checking_mutex> lock(mux);
  assert(mux.current_state == checking_mutex::locked_via_lock);
  lock.unlock();
  assert(mux.current_state == checking_mutex::unlocked);
  assert(!lock.owns_lock());

#ifndef TEST_HAS_NO_EXCEPTIONS
  try {
    mux.last_try = checking_mutex::none;
    lock.unlock();
    assert(false);
  } catch (::std::system_error& e) {
    assert(mux.last_try == checking_mutex::none);
    assert(e.code() == ::std::errc::operation_not_permitted);
  }
#endif

  lock.release();

#ifndef TEST_HAS_NO_EXCEPTIONS
  try {
    mux.last_try = checking_mutex::none;
    lock.unlock();
    assert(false);
  } catch (::std::system_error& e) {
    assert(mux.last_try == checking_mutex::none);
    assert(e.code() == ::std::errc::operation_not_permitted);
  }
#endif
#endif

  return 0;
}

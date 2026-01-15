//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <mutex>

// template <class Mutex> class unique_lock;

// unique_lock(mutex_type& m, defer_lock_t);

// ADDITIONAL_COMPILE_FLAGS: -DTEST_NO_HIP_THREAD

#include <cassert>
#include <memory>
#include <mutex>
#include <type_traits>
#include <hip/mutex>

#include "force_include_hip.h"

#include "checking_mutex.h"
#include "test_macros.h"

#if TEST_STD_VER >= 11
static_assert(
    ::std::is_nothrow_constructible<hip::unique_lock<checking_mutex>, checking_mutex&, ::std::defer_lock_t>::value, "");
#endif

int main(int, char**) {
#ifdef __HIP_DEVICE_COMPILE__
  checking_mutex m;
  hip::unique_lock<checking_mutex> lk(m, ::std::defer_lock_t());
  assert(m.last_try == checking_mutex::none);
  assert(lk.mutex() == hip::std::addressof(m));
  assert(lk.owns_lock() == false);
#endif

  return 0;
}

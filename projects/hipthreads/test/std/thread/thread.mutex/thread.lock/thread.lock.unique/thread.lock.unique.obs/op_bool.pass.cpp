//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <mutex>

// template <class Mutex> class unique_lock;

// explicit operator bool() const noexcept;

// ADDITIONAL_COMPILE_FLAGS: -DTEST_NO_HIP_THREAD

#include <cassert>
#include <type_traits>
#include <hip/mutex>

#include "force_include_hip.h"

#include "checking_mutex.h"
#include "test_macros.h"

#if TEST_STD_VER >= 11
static_assert(noexcept(static_cast<bool>(::std::declval<hip::unique_lock<checking_mutex>&>())), "");
#endif

int main(int, char**) {
#ifdef __HIP_DEVICE_COMPILE__
  static_assert(::std::is_constructible<bool, hip::unique_lock<checking_mutex> >::value, "");
  static_assert(!::std::is_convertible<hip::unique_lock<checking_mutex>, bool>::value, "");

  checking_mutex mux;
  const hip::unique_lock<checking_mutex> lk0; // Make sure `operator bool()` is `const`
  assert(!static_cast<bool>(lk0));
  hip::unique_lock<checking_mutex> lk1(mux);
  assert(static_cast<bool>(lk1));
  lk1.unlock();
  assert(!static_cast<bool>(lk1));

  ASSERT_NOEXCEPT(static_cast<bool>(lk0));
#endif

  return 0;
}

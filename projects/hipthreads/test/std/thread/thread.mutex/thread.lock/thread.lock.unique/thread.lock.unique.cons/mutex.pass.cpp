//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <mutex>

// template <class Mutex> class unique_lock;

// explicit unique_lock(mutex_type& m);

// template<class _Mutex> unique_lock(unique_lock<_Mutex>)
//     -> unique_lock<_Mutex>;  // C++17

// ADDITIONAL_COMPILE_FLAGS: -DTEST_NO_HIP_THREAD

#include <cassert>
#include <hip/mutex>

#include "force_include_hip.h"

#include "checking_mutex.h"
#include "test_macros.h"

int main(int, char**) {
#ifdef __HIP_DEVICE_COMPILE__
  checking_mutex mux;

  {
    hip::unique_lock<checking_mutex> lock(mux);
    assert(mux.current_state == checking_mutex::locked_via_lock);
  }
  assert(mux.current_state == checking_mutex::unlocked);

#if TEST_STD_VER >= 17
  static_assert(::std::is_same_v<hip::unique_lock<checking_mutex>, decltype(hip::unique_lock{mux})>, "");
#endif
#endif

  return 0;
}

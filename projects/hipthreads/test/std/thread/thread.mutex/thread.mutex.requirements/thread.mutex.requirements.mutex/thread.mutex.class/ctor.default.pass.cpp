//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads
// ADDITIONAL_COMPILE_FLAGS: -DTEST_NO_HIP_THREAD

// <mutex>

// class mutex;

// mutex() noexcept;

#include <hip/mutex>
#include <cassert>
#include <type_traits>

#include "force_include_hip.h"

static_assert(::std::is_nothrow_default_constructible<::std::mutex>::value, "");

int main(int, char**) {
#ifdef __HIP_DEVICE_COMPILE__
  // The mutex is unlocked after default construction
  {
    hip::spin_mutex m;
    assert(m.try_lock());
    m.unlock();
  }
#endif

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03

// Test the interoperation of hip::lock_guard with ::std::mutex, since that is such
// a common use case.

#include <cassert>
#include <mutex>
#include <type_traits>
#include <functional>
#include <hip/mutex>

#include "force_include_hip.h"

#include "make_test_thread.h"
#include "test_macros.h"

__device__ void do_try_lock(hip::spin_mutex& m) { assert(m.try_lock() == false); }

int main(int, char**) {
#ifdef __HIP_DEVICE_COMPILE
  {
    hip::spin_mutex m;
    {
      hip::lock_guard<hip::spin_mutex> lg(m);
      hip::wthread t = support::make_test_thread(do_try_lock, ::std::ref(m));
      t.join();
    }

    // This should work because the lock_guard unlocked the mutex when it was destroyed above.
    m.lock();
    m.unlock();
  }

  // Test CTAD
#if TEST_STD_VER >= 17
  {
    hip::spin_mutex m;
    hip::lock_guard lg(m);
    static_assert(::std::is_same<decltype(lg), hip::lock_guard<hip::spin_mutex>>::value, "");
  }
#endif
#endif

  return 0;
}

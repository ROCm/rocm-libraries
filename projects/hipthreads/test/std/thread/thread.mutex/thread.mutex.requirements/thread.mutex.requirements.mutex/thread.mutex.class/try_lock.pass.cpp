//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// UNSUPPORTED: no-threads

// <mutex>

// class mutex;

// bool try_lock();

#include <cassert>
#include <hip/mutex>
#include <hip/thread>
#include <hip/std/memory>

#include "force_include_hip.h"

#include "make_test_thread.h"

int main(int, char**) {
#ifdef __HIP_DEVICE_COMPILE__
  // Try to lock a mutex that is not locked yet. This should succeed.
  {
    hip::spin_mutex m;
    bool succeeded = m.try_lock();
    assert(succeeded);
    m.unlock();
  }

  // Try to lock a mutex that is already locked. This should fail.
  {
    auto m_ptr = hip::std::make_unique<hip::spin_mutex>();
    hip::spin_mutex &m = *m_ptr;
    m.lock();

    hip::wthread t = support::make_test_thread([&] {
      for (int i = 0; i != 10; ++i) {
        bool succeeded = m.try_lock();
        assert(!succeeded);
      }
    });
    t.join();

    m.unlock();
  }
#endif

  return 0;
}

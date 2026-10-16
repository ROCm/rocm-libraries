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

// void lock();

#include <cassert>
#include <hip/atomic>
#include <hip/mutex>
#include <hip/thread>
#include <hip/std/inplace_vector>
#include <hip/std/memory>

#include "force_include_hip.h"

#include "make_test_thread.h"

int main(int, char**) {
#ifdef __HIP_DEVICE_COMPILE__
  // Lock a mutex that is not locked yet. This should succeed.
  {
    hip::spin_mutex m;
    m.lock();
    m.unlock();
  }

  // Lock a mutex that is already locked. This should block until it is unlocked.
  {
    auto ready_ptr = hip::std::make_unique<hip::std::atomic<bool>>(false);
    auto m_ptr = hip::std::make_unique<hip::spin_mutex>();
    auto is_locked_from_main_ptr = hip::std::make_unique<hip::std::atomic<bool>>(true);
    hip::std::atomic<bool> &ready = *ready_ptr;
    hip::spin_mutex &m = *m_ptr;
    hip::std::atomic<bool> &is_locked_from_main = *is_locked_from_main_ptr;

    m.lock();

    hip::wthread t = support::make_test_thread([&] {
      ready = true;
      m.lock();
      assert(!is_locked_from_main);
      m.unlock();
    });

    while (!ready)
      /* spin */;

    // We would rather signal this after we unlock, but that would create a race condition.
    // We instead signal it before we unlock, which means that it's technically possible for
    // the thread to take the lock while main is still holding it yet for the test to still pass.
    is_locked_from_main = false;
    m.unlock();

    t.join();
  }

  // Make sure that at most one thread can acquire the mutex concurrently.
  {
    auto counter_ptr = hip::std::make_unique<hip::std::atomic<int>>(0);
    auto mutex_ptr = hip::std::make_unique<hip::spin_mutex>();
    hip::std::atomic<int> &counter = *counter_ptr;
    hip::spin_mutex &mutex = *mutex_ptr;

    hip::std::inplace_vector<hip::wthread, 10> threads;
    for (int i = 0; i != 10; ++i) {
      threads.push_back(support::make_test_thread([&] {
        mutex.lock();
        counter++;
        assert(counter == 1);
        counter--;
        mutex.unlock();
      }));
    }

    for (auto& t : threads)
      t.join();
  }
#endif

  return 0;
}

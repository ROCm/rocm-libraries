//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads, c++03
// <condition_variable>

// class condition_variable_any;

// template <class Lock, class Predicate>
//   void wait(Lock& lock, Predicate pred);

#include <cassert>
#include <hip/atomic>
#include <hip/condition_variable>
#include <hip/mutex>
#include <hip/thread>
#include <hip/std/memory>

#include "make_test_thread.h"
#include "test_macros.h"

#include "force_include_hip.h"

template <class Mutex>
struct MyLock : hip::unique_lock<Mutex> {
  using hip::unique_lock<Mutex>::unique_lock;
};

template <class Lock>
__device__ void test() {
  using Mutex = typename Lock::mutex_type;

  // Test unblocking via a call to notify_one() in another thread.
  //
  // To test this, we try to minimize the likelihood that we got awoken by a
  // spurious wakeup by updating the likely_spurious flag only immediately
  // before we perform the notification.
  {
    auto ready_ptr = hip::std::make_unique<hip::std::atomic<bool>>(false);
    auto likely_spurious_ptr = hip::std::make_unique<hip::std::atomic<bool>>(true);
    auto cv_ptr = hip::std::make_unique<hip::condition_variable_any>();
    auto mutex_ptr = hip::std::make_unique<Mutex>();
    hip::std::atomic<bool> &ready = *ready_ptr;
    hip::std::atomic<bool> &likely_spurious = *likely_spurious_ptr;
    hip::condition_variable_any &cv = *cv_ptr;
    Mutex &mutex = *mutex_ptr;

    hip::wthread t1 = support::make_test_thread([&] {
      Lock lock(mutex);
      ready = true;
      cv.wait(lock, [&] { return !likely_spurious; });
    });

    hip::wthread t2 = support::make_test_thread([&] {
      while (!ready) {
        // spin
      }

      // Acquire the same mutex as t1. This ensures that the condition variable has started
      // waiting (and hence released that mutex).
      Lock lock(mutex);

      likely_spurious = false;
      lock.unlock();
      cv.notify_one();
    });
    
    t2.join();
    t1.join();
  }

  // Test unblocking via a spurious wakeup.
  //
  // To test this, we basically never wake up the condition variable. This way, we
  // are hoping to get out of the wait via a spurious wakeup.
  //
  // However, since spurious wakeups are not required to even happen, this test is
  // only trying to trigger that code path, but not actually asserting that it is
  // taken. In particular, we do need to eventually ensure we get out of the wait
  // by standard means, so we actually wake up the thread at the end.
  {
    auto ready_ptr = hip::std::make_unique<hip::std::atomic<bool>>(false);
    auto awoken_ptr = hip::std::make_unique<hip::std::atomic<bool>>(false);
    auto cv_ptr = hip::std::make_unique<hip::condition_variable_any>();
    auto mutex_ptr = hip::std::make_unique<Mutex>();
    hip::std::atomic<bool> &ready = *ready_ptr;
    hip::std::atomic<bool> &awoken = *awoken_ptr;
    hip::condition_variable_any &cv = *cv_ptr;
    Mutex &mutex = *mutex_ptr;
    
    hip::wthread t1 = support::make_test_thread([&] {
      Lock lock(mutex);
      ready = true;
      cv.wait(lock, [] { return true; });
      awoken = true;
    });

    hip::wthread t2 = support::make_test_thread([&] {
      while (!ready) {
        // spin
      }

      // Acquire the same mutex as t1. This ensures that the condition variable has started
      // waiting (and hence released that mutex).
      Lock lock(mutex);
      lock.unlock();

      // Give some time for t1 to be awoken spuriously so that code path is used.
      hip::this_thread::sleep_for(cuda::std::chrono::seconds(1));

      // We would want to assert that the thread has been awoken after this time,
      // however nothing guarantees us that it ever gets spuriously awoken, so
      // we can't really check anything. This is still left here as documentation.
      bool woke = awoken.load();
      assert(woke || !woke);

      // Whatever happened, actually awaken the condition variable to ensure the test finishes.
      cv.notify_one();
    });
    

    t2.join();
    t1.join();
  }
}

int main(int, char**) {
#ifdef __HIP_DEVICE_COMPILE__
  test<hip::unique_lock<hip::spin_mutex>>();
  test<MyLock<hip::spin_mutex>>();
  // TODO: re-add testing using hip::timed_spin_mutex once we support it
  // test<hip::unique_lock<hip::timed_spin_mutex>>();
  // test<MyLock<hip::timed_spin_mutex>>();
#endif

  return 0;
}

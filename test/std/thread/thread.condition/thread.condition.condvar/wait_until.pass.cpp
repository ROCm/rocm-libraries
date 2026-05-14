//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads, c++03

// <condition_variable>

// class condition_variable;

// template <class Clock, class Duration>
//   cv_status
//   wait_until(unique_lock<mutex>& lock,
//              const chrono::time_point<Clock, Duration>& abs_time);

#include <cassert>
#include <hip/atomic>
#include <hip/std/chrono>
#include <hip/condition_variable>
#include <hip/mutex>
#include <hip/thread>

#include "make_test_thread.h"
#include "test_macros.h"

#include "force_include_hip.h"

struct TestClock {
  typedef hip::std::chrono::milliseconds duration;
  typedef duration::rep rep;
  typedef duration::period period;
  typedef hip::std::chrono::time_point<TestClock> time_point;
  static const bool is_steady = true;

  __device__ static time_point now() {
    using namespace hip::std::chrono;
    return time_point(duration_cast<duration>(system_clock::now().time_since_epoch()));
  }
};

template <class Clock>
__device__ void test() {
  printf("test<Clock> running on device: block=%d thread=%d\n", blockIdx.x, threadIdx.x);
  // Test unblocking via a call to notify_one() in another thread.
  //
  // To test this, we set a very long timeout in wait_until() and we wait
  // again in case we get awoken spuriously. Note that it can actually
  // happen that we get awoken spuriously and fail to recognize it
  // (making this test useless), but the likelihood should be small.
  {
    hip::atomic<bool> ready(false);
    hip::atomic<bool> likely_spurious(true);
    auto timeout = Clock::now() + hip::std::chrono::seconds(3600);
    hip::spin_condition_variable cv;
    hip::spin_mutex mutex;

    hip::thread t1 = support::make_test_thread([&] {
      printf("t1: block=%d thread=%d\n", blockIdx.x, threadIdx.x);
      printf("  t1: start\n");
      hip::unique_lock<hip::spin_mutex> lock(mutex);
      printf("  t1: locked, setting ready=true\n");
      ready = true;
      do {
        ::std::cv_status result = cv.wait_until(lock, timeout);
        assert(result == ::std::cv_status::no_timeout);
      } while (likely_spurious);
      // This can technically fail if we have many spurious awakenings, but in practice the
      // tolerance is so high that it shouldn't be a problem.
      assert(Clock::now() < timeout);
      printf("  t1: end\n");
    });

    hip::thread t2 = support::make_test_thread([&] {
      printf("t2: block=%d thread=%d\n", blockIdx.x, threadIdx.x);
      printf("  t2: start, spinning for ready\n");
      while (!ready) {
        // spin
      }
      printf("  t2: ready seen, locking\n");

      // Acquire the same mutex as t1. This blocks the condition variable inside its wait call
      // so we can notify it while it is waiting.
      hip::unique_lock<hip::spin_mutex> lock(mutex);
      printf("  t2: locked, notifying\n");
      likely_spurious = false;
      cv.notify_one();
    
      lock.unlock();
      printf("  t2: end\n");
    });

    printf("parent: spawned, joining t2\n");
    t2.join();
    printf("parent: t2 joined, joining t1\n");
    t1.join();
    printf("parent: both joined\n");
  }

  // Test unblocking via a timeout.
  //
  // To test this, we create a thread that waits on a condition variable
  // with a certain timeout, and we never awaken it. To guard against
  // spurious wakeups, we wait again whenever we are awoken for a reason
  // other than a timeout.
  {
    auto timeout = Clock::now() + hip::std::chrono::milliseconds(250);
    hip::spin_condition_variable cv;
    hip::spin_mutex mutex;

    hip::thread t1 = support::make_test_thread([&] {
      hip::unique_lock<hip::spin_mutex> lock(mutex);
      ::std::cv_status result;
      do {
        result = cv.wait_until(lock, timeout);
        if (result == ::std::cv_status::timeout)
          assert(Clock::now() >= timeout);
      } while (result != ::std::cv_status::timeout);
    });

    t1.join();
  }
}

int main(int, char**) {
#ifdef __HIP_DEVICE_COMPILE__
  test<TestClock>();
  test<hip::std::chrono::system_clock>(); 
#endif
  return 0;
}

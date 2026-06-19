//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: availability-synchronization_library-missing

// void join();

#include <cassert>
#include <concepts>
#include <functional>
#include <hip/atomic>
#include <hip/std/chrono>
#include <hip/std/memory>
#include <hip/thread>
#include <system_error>
#include <type_traits>

#include "force_include_hip.h"
#include "make_test_thread.h"
#include "test_macros.h"

int main(int, char**) {
#ifdef __HIP_DEVICE_COMPILE__
  // Effects: Blocks until the thread represented by *this has completed.
  {
    auto calledTimes_ptr = hip::std::make_unique<hip::std::atomic<int>>(0);
    auto& calledTimes    = *calledTimes_ptr;
    constexpr auto numberOfThreads = 10u;
    hip::jthread jts[numberOfThreads];
    for (auto i = 0u; i < numberOfThreads; ++i) {
      jts[i] = support::make_test_jthread([&calledTimes] () {
        hip::this_thread::sleep_for(hip::std::chrono::milliseconds(2));
        calledTimes.fetch_add(1, hip::std::memory_order_relaxed);
      });
    }

    for (auto i = 0u; i < numberOfThreads; ++i) {
      jts[i].join();
    }

    // If join did block, calledTimes must equal to numberOfThreads
    // If join did not block, there is a chance that the check below happened
    // before test threads incrementing the counter, thus calledTimed would
    // be less than numberOfThreads.
    // This is not going to catch issues 100%. Creating more threads to increase
    // the probability of catching the issue
    assert(calledTimes.load(hip::std::memory_order_relaxed) == numberOfThreads);
  }

  // Synchronization: The completion of the thread represented by *this synchronizes with
  // ([intro.multithread]) the corresponding successful join() return.
  {
    auto flag_ptr = hip::std::make_unique<bool>(false);
    auto& flag    = *flag_ptr;
    hip::jthread jt = support::make_test_jthread([&flag] () { flag = true; });
    jt.join();
    assert(flag); // non atomic write is visible to the current thread
  }

  // Postconditions: The thread represented by *this has completed. get_id() == id().
  {
    hip::jthread jt = support::make_test_jthread([] () {});
    assert(jt.get_id() != hip::jthread::id());
    jt.join();
    assert(jt.get_id() == hip::jthread::id());
  }

#if !defined(TEST_HAS_NO_EXCEPTIONS)
  // Throws: system_error when an exception is required ([thread.req.exception]).
  // invalid_argument - if the thread is not joinable.
  {
    hip::jthread jt;
    try {
      jt.join();
      assert(false);
    } catch (const ::std::system_error& err) {
      assert(err.code() == ::std::errc::invalid_argument);
    }
  }

#endif
#endif
  return 0;
}

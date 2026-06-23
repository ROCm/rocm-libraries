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

// ~jthread();

#include <cassert>
#include <hip/atomic>
#include <hip/std/chrono>
#include <hip/std/inplace_vector>
#include <hip/std/memory>
#include <hip/thread>
#include <type_traits>

#include "force_include_hip.h"
#include "make_test_thread.h"
#include "test_macros.h"

int main(int, char**) {
  // !joinable() — default-constructed jthread is not joinable.
  {
    hip::jthread jt;
    assert(!jt.joinable());
  }

  // TODO: Divergence from std::jthread: the upstream test block that verifies
  // request_stop() and stop_callback firing on destruction is omitted because
  // hip::jthread has no stop-token support (deliberate; future work).

#ifdef __HIP_DEVICE_COMPILE__
  // If joinable() is true, the destructor calls join().
  // Spawn several jthreads, let them increment a shared counter, then destroy
  // the array. If auto-join works, all increments must be visible after
  // destruction.
  {
    constexpr auto numberOfThreads = 10u;

    // Heap-allocated atomic shared with the device lambdas via reference —
    // see project convention for cross-block stack access on GPU.
    auto calledTimes_ptr = hip::std::make_unique<hip::std::atomic<int>>(0);
    auto& calledTimes    = *calledTimes_ptr;

    hip::std::inplace_vector<hip::jthread, numberOfThreads> jts;
    for (auto i = 0u; i < numberOfThreads; ++i) {
      jts.emplace_back(support::make_test_jthread([&calledTimes] {
        hip::this_thread::sleep_for(hip::std::chrono::milliseconds{2});
        calledTimes.fetch_add(1, hip::std::memory_order_relaxed);
      }));
    }
    jts.clear();  // ~jthread() runs here for every element → auto-join

    // If join was called as expected, calledTimes must equal numberOfThreads.
    // If join was not called, the assert below could race with the worker
    // threads incrementing the counter; observing the full count here proves
    // the destructor synchronously joined each worker before returning.
    assert(calledTimes.load(hip::std::memory_order_relaxed) == numberOfThreads);
  }
#endif

  return 0;
}

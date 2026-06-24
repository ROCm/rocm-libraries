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
// ADDITIONAL_COMPILE_FLAGS(gcc-style-warnings): -Wno-self-move

// jthread& operator=(jthread&&) noexcept;

#include <cassert>
#include <concepts>
#include <hip/atomic>
#include <hip/std/chrono>
#include <hip/std/inplace_vector>
#include <hip/std/memory>
#include <hip/thread>
#include <type_traits>
#include <utility>

#include "force_include_hip.h"
#include "make_test_thread.h"
#include "test_macros.h"

static_assert(::std::is_nothrow_move_assignable_v<hip::jthread>);

int main(int, char**) {
#ifdef __HIP_DEVICE_COMPILE__
  // If &x == this is true, there are no effects.
  {
    hip::jthread j = support::make_test_jthread([] __device__ () {});
    auto id        = j.get_id();
    // TODO: stop token not implemented
    // auto ssource = j.get_stop_source();
    j              = ::std::move(j);
    assert(j.get_id() == id);
    // TODO: stop token not implemented
    // assert(j.get_stop_source() == ssource);
  }

  // if joinable() is true, calls request_stop() and then join()
  // request_stop is called
  // TODO: stop token not implemented
  // {
  //   hip::jthread j1 = support::make_test_jthread([] __device__ () {});
  //   bool called     = false;
  //   hip::stop_callback cb(j1.get_stop_token(), [&called] { called = true; });
  //
  //   hip::jthread j2 = support::make_test_jthread([] __device__ () {});
  //   j1              = ::std::move(j2);
  //   assert(called);
  // }

  // if joinable() is true, calls request_stop() and then join()
  // join is called
  {
    auto calledTimes_ptr = hip::std::make_unique<hip::std::atomic<int>>(0);
    auto& calledTimes    = *calledTimes_ptr;
    constexpr auto numberOfThreads = 10u;
    hip::std::inplace_vector<hip::jthread, numberOfThreads> jts;
    for (auto i = 0u; i < numberOfThreads; ++i) {
      jts.emplace_back(support::make_test_jthread([&] __device__ () {
        hip::this_thread::sleep_for(cuda::std::chrono::milliseconds(2));
        calledTimes.fetch_add(1, hip::std::memory_order_relaxed);
      }));
    }

    for (auto i = 0u; i < numberOfThreads; ++i) {
      jts[i] = hip::jthread{};
    }

    // If join was called as expected, calledTimes must equal to numberOfThreads
    // If join was not called, there is a chance that the check below happened
    // before test threads incrementing the counter, thus calledTimed would
    // be less than numberOfThreads.
    // This is not going to catch issues 100%. Creating more threads to increase
    // the probability of catching the issue
    assert(calledTimes.load(hip::std::memory_order_relaxed) == numberOfThreads);
  }

  // then assigns the state of x to *this
  {
    hip::jthread j1 = support::make_test_jthread([] __device__ () {});
    hip::jthread j2 = support::make_test_jthread([] __device__ () {});
    auto id2        = j2.get_id();
    // TODO: stop token not implemented
    // auto ssource2 = j2.get_stop_source();

    j1 = ::std::move(j2);

    assert(j1.get_id() == id2);
    // TODO: stop token not implemented
    // assert(j1.get_stop_source() == ssource2);
  }

  // sets x to a default constructed state
  {
    hip::jthread j1 = support::make_test_jthread([] __device__ () {});
    hip::jthread j2 = support::make_test_jthread([] __device__ () {});
    j1              = ::std::move(j2);

    assert(j2.get_id() == hip::jthread::id());
    // TODO: stop token not implemented
    // assert(!j2.get_stop_source().stop_possible());
  }

  // joinable is false
  {
    hip::jthread j1;
    hip::jthread j2 = support::make_test_jthread([] __device__ () {});

    auto j2Id = j2.get_id();

    j1 = ::std::move(j2);

    assert(j1.get_id() == j2Id);
  }
#endif
  return 0;
}

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

// void detach();

#include <cassert>
#include <concepts>
#include <hip/atomic>
#include <hip/std/chrono>
#include <hip/std/memory>
#include <hip/std/optional>
#include <hip/thread>
#include <type_traits>

#include "force_include_hip.h"
#include "make_test_thread.h"
#include "test_macros.h"

int main(int, char**) {
#ifdef __HIP_DEVICE_COMPILE__
  // Effects: The thread represented by *this continues execution without the calling thread blocking.
  {
    auto start_ptr = hip::std::make_unique<hip::std::atomic<bool>>(false);
    auto done_ptr  = hip::std::make_unique<hip::std::atomic<bool>>(false);
    auto& start    = *start_ptr;
    auto& done     = *done_ptr;

    hip::std::optional<hip::jthread> jt = support::make_test_jthread([&start, &done] __device__ () {
      hip::std::atomic_wait(&start, false);
      done = true;
    });

    // If it blocks, it will deadlock here
    jt->detach();

    jt.reset();

    // The other thread continues execution
    start = true;
    while (!done) {
    }
  }

  // Postconditions: get_id() == id().
  {
    hip::jthread jt = support::make_test_jthread([] __device__ () {});
    assert(jt.get_id() != hip::jthread::id());
    jt.detach();
    assert(jt.get_id() == hip::jthread::id());
  }

#if !defined(TEST_HAS_NO_EXCEPTIONS)
  // Throws: system_error when an exception is required ([thread.req.exception]).
  // invalid_argument - if the thread is not joinable.
  {
    hip::jthread jt;
    try {
      jt.detach();
      assert(false);
    } catch (const ::std::system_error& err) {
      assert(err.code() == ::std::errc::invalid_argument);
    }
  }
#endif

  hip::this_thread::sleep_for(hip::std::chrono::milliseconds{2});
#endif
  return 0;
}

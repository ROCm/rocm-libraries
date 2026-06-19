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

// [[nodiscard]] bool joinable() const noexcept;

#include <cassert>
#include <concepts>
#include <hip/atomic>
#include <hip/thread>
#include <hip/std/memory>
#include <type_traits>

#include "force_include_hip.h"
#include "make_test_thread.h"
#include "test_macros.h"

static_assert(noexcept(::std::declval<const hip::jthread&>().joinable()));

int main(int, char**) {
#ifdef __HIP_DEVICE_COMPILE__   
  // Default constructed
  {
    const hip::jthread jt;
    ::std::same_as<bool> decltype(auto) result = jt.joinable();
    assert(!result);
  }

  // Non-default constructed
  {
    const hip::jthread jt                      = support::make_test_jthread([] () {});
    ::std::same_as<bool> decltype(auto) result = jt.joinable();
    assert(result);
  }

  // Non-default constructed
  // the thread of execution has not finished
  {
    auto done_ptr                = hip::std::make_unique<hip::std::atomic<bool>>(false);
    hip::std::atomic<bool>& done = *done_ptr;
    const hip::jthread jt        = support::make_test_jthread([&done] {
      hip::std::atomic_wait(&done, false);
    });
    ::std::same_as<bool> decltype(auto) result = jt.joinable();
    done                                       = true;
    assert(result);
  }
#endif  
  return 0;
}

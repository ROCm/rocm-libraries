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

// [[nodiscard]] id get_id() const noexcept;

#include <cassert>
#include <concepts>
#include <hip/thread>
#include <type_traits>

#include "force_include_hip.h"
#include "make_test_thread.h"
#include "test_macros.h"

// Divergence from std::jthread::get_id(): hip::jthread::get_id forwards to
// hip::thread::get_id(uint32_t index), which can throw std::out_of_range on host
// when index >= width. Marking it noexcept would convert such throws into
// std::terminate, so we intentionally diverge and assert non-noexcept here.
static_assert(!noexcept(::std::declval<const hip::jthread&>().get_id()));

int main(int, char**) {
#ifdef __HIP_DEVICE_COMPILE__  
  // Does not represent a thread
  {
    const hip::jthread jt;
    ::std::same_as<hip::jthread::id> decltype(auto) result = jt.get_id();
    assert(result == hip::jthread::id());
  }
  // Represents a thread
  {
    const hip::jthread jt                                = support::make_test_jthread([] __device__{});
    ::std::same_as<hip::jthread::id> decltype(auto) result = jt.get_id();
    assert(result != hip::jthread::id());
  }
#endif  
  return 0;
}

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

// jthread(jthread&& x) noexcept;

#include <cassert>
#include <hip/thread>
#include <type_traits>
#include <utility>

#include "force_include_hip.h"
#include "make_test_thread.h"
#include "test_macros.h"

static_assert(::std::is_nothrow_move_constructible_v<hip::jthread>);

int main(int, char**) {
#ifdef __HIP_DEVICE_COMPILE__
  {
    // x.get_id() == id() and get_id() returns the value of x.get_id() prior
    // to the start of construction.
    hip::jthread j1 = support::make_test_jthread([] () {});
    auto id1        = j1.get_id();

    hip::jthread j2(::std::move(j1));
    assert(j1.get_id() == hip::jthread::id());
    assert(j2.get_id() == id1);
  }

  // {
  //   // ssource has the value of x.ssource prior to the start of construction
  //   // and x.ssource.stop_possible() is false.
  //   hip::jthread j1 = support::make_test_jthread([] () {});
  //   auto ss1        = j1.get_stop_source();
  //
  //   hip::jthread j2(::std::move(j1));
  //   assert(ss1 == j2.get_stop_source());
  //   assert(!j1.get_stop_source().stop_possible());
  //   assert(j2.get_stop_source().stop_possible());
  // } // stop token not implemented
#endif
  return 0;
}

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

// void swap(jthread& x) noexcept;

#include <cassert>
#include <hip/thread>
#include <type_traits>

#include "force_include_hip.h"
#include "make_test_thread.h"
#include "test_macros.h"

template <class T>
concept IsMemberSwapNoexcept = requires(T& a, T& b) {
  { a.swap(b) } noexcept;
};

static_assert(IsMemberSwapNoexcept<hip::jthread>);

int main(int, char**) {
#ifdef __HIP_DEVICE_COMPILE__
  // this is default constructed
  {
    hip::jthread t1;
    hip::jthread t2        = support::make_test_jthread([] __device__ () {});
    const auto originalId2 = t2.get_id();
    t1.swap(t2);

    assert(t1.get_id() == originalId2);
    assert(t2.get_id() == hip::jthread::id());
  }

  // that is default constructed
  {
    hip::jthread t1 = support::make_test_jthread([] __device__ () {});
    hip::jthread t2{};
    const auto originalId1 = t1.get_id();
    t1.swap(t2);

    assert(t1.get_id() == hip::jthread::id());
    assert(t2.get_id() == originalId1);
  }

  // both not default constructed
  {
    hip::jthread t1        = support::make_test_jthread([] __device__ () {});
    hip::jthread t2        = support::make_test_jthread([] __device__ () {});
    const auto originalId1 = t1.get_id();
    const auto originalId2 = t2.get_id();
    t1.swap(t2);

    assert(t1.get_id() == originalId2);
    assert(t2.get_id() == originalId1);
  }

  // both default constructed
  {
    hip::jthread t1;
    hip::jthread t2;
    t1.swap(t2);

    assert(t1.get_id() == hip::jthread::id());
    assert(t2.get_id() == hip::jthread::id());
  }
#endif
  return 0;
}

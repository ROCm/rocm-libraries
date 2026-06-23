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

// jthread() noexcept;

#include <cassert>
#include <hip/thread>
#include <type_traits>

#include "force_include_hip.h"
#include "test_macros.h"

static_assert(::std::is_nothrow_default_constructible_v<hip::jthread>);

int main(int, char**) {
  {
    hip::jthread jt = {}; // implicit
    // TODO: stop token not implemented
    // assert(!jt.get_stop_source().stop_possible());
    assert(jt.get_id() == hip::jthread::id());
  }

  return 0;
}

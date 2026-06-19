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
// [[nodiscard]] id get_id() const noexcept;
// [[nodiscard]] static unsigned int hardware_concurrency() noexcept;
//
// Divergence from std::jthread: hip::jthread does not expose native_handle(),
// get_stop_source(), or get_stop_token() (no stop-token support yet), so the
// corresponding nodiscard checks from upstream libcxx are commented out.

#include <hip/thread>
// libhipcxx emits a #warning about TSC clock rate during device compile for
// some gfx targets; whitelist it so -verify doesn't fail on this unrelated
// environmental noise.
// expected-warning@*:* 0+ {{realtime clock rate}}

void test() {
  hip::jthread jt;
  jt.joinable();             // expected-warning {{ignoring return value of function}}
  jt.get_id();               // expected-warning {{ignoring return value of function}}
  // Lines below are commented out because hip::jthread does not expose these
  // methods (no native_handle / stop-token support yet). The "expected-warning"
  // keyword is intentionally removed so clang -verify does not register stale
  // directives that would never fire.
  // jt.native_handle();        -- not exposed by hip::jthread
  // jt.get_stop_source();      -- not exposed by hip::jthread
  // jt.get_stop_token();       -- not exposed by hip::jthread
  jt.hardware_concurrency(); // expected-warning {{ignoring return value of function}}
}

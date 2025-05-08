//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <atomic>

// template <class T>
// struct atomic;

// This test checks that we static_assert inside gpu::atomic<T> when T
// is not trivially copyable, however Clang will sometimes emit additional
// errors while trying to instantiate the rest of gpu::atomic<T>.
// We silence those to make the test more robust.
// ADDITIONAL_COMPILE_FLAGS: -Xclang -verify-ignore-unexpected=error

#include "gpu/atomic"
#include "hip/hip_runtime.h"

struct NotTriviallyCopyable {
  __device__ explicit NotTriviallyCopyable(int i) : i_(i) { }
  __device__ NotTriviallyCopyable(const NotTriviallyCopyable &rhs) : i_(rhs.i_) { }
  int i_;
};

__global__ void f() {
  NotTriviallyCopyable x(42);
  gpu::atomic<NotTriviallyCopyable> a(x); // expected-error@*:* {{std::atomic<T> requires that 'T' be a trivially copyable type}}
}

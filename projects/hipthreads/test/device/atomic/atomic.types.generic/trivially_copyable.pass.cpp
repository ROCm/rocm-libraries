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

// Make sure atomic<TriviallyCopyable> can be instantiated.

#include "gpu/atomic"
#include <new>
#include <cassert>
#include <chrono> // for nanoseconds

#include "test_macros.h"
#include "kernel_launcher.h"

#ifndef TEST_HAS_NO_THREADS
#  include <thread> // for thread_id
#endif

struct TriviallyCopyable {
  __device__ explicit TriviallyCopyable(int i) : i_(i) { }
  int i_;
};

template <class T>
__device__ void test(T t) {
  gpu::atomic<T> t0(t);
}

__global__ void gmain() {
  test(TriviallyCopyable(42));
  test(std::chrono::nanoseconds(2));
#ifndef TEST_HAS_NO_THREADS
  // TODO: Uncomment this and use gpu::this_thread::get_id()
  //test(std::this_thread::get_id());
#endif
}

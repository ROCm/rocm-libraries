//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-exceptions

// <atomic>

#include "gpu/atomic"
#include <cassert>

#include "kernel_launcher.h"

struct throwing {
  __device__ throwing() { throw 42; }
};

__global__ void gmain() {
  try {
    [[maybe_unused]] gpu::atomic<throwing> a;
    assert(false);
  } catch (int x) {
    assert(x == 42);
  }
}

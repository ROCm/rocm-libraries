//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03

// hip::threads doesn't currently support constructing threads during the destruction of global variables
// XFAIL: *

#include "make_test_thread.h"

__device__ void func() {}

struct T {
  ~T() {
    // __thread_local_data is expected to be destroyed as it was created
    // from the main(). Now trigger another access.
    support::make_test_thread([] __device__(){func();}).join();
  }
} t;
// __device__ T t2;

int main(int, char**) {
  // Triggers construction of __thread_local_data.
  support::make_test_thread([] __device__(){func();}).join();

  return 0;
}

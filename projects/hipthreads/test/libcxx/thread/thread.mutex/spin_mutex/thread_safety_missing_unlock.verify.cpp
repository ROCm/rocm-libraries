//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// On Windows Clang bugs out when both __declspec and __attribute__ are present,
// the processing goes awry preventing the definition of the types.
// XFAIL: msvc

// UNSUPPORTED: no-threads
// REQUIRES: thread-safety
// ADDITIONAL_COMPILE_FLAGS: -DTEST_NO_HIP_THREAD

// <mutex>

// ADDITIONAL_COMPILE_FLAGS: -D_LIBHIPTHREADS_ENABLE_THREAD_SAFETY_ANNOTATIONS

#include <hip/mutex>

__device__ hip::spin_mutex m;

__device__ void f() {
  m.lock();
} // expected-error {{mutex 'm' is still held at the end of function}}

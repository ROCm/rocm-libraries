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

#include <hip/pseudo_mutex>

#include "test_macros.h"

#include "force_include_hip.h"

__device__ hip::pseudo_mutex m;
__device__ int foo __attribute__((guarded_by(m)));

int main(int, char**) {
#ifdef __HIP_DEVICE_COMPILE__
  m.lock();
  foo++;
  m.unlock();
#endif

  return 0;
}

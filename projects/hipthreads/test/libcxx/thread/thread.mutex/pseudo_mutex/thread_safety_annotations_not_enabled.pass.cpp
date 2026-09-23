//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03
// ADDITIONAL_COMPILE_FLAGS: -DTEST_NO_HIP_THREAD

// <mutex>

// This test does not define _LIBCPP_ENABLE_THREAD_SAFETY_ANNOTATIONS so it
// should compile without any warnings or errors even though this pattern is not
// understood by the thread safety annotations.

#include <hip/pseudo_mutex>
#include <hip/mutex>

#include "test_macros.h"

#include "force_include_hip.h"

int main(int, char**) {
#ifdef __HIP_DEVICE_COMPILE__
  hip::pseudo_mutex m;
  m.lock();
  {
    hip::unique_lock<hip::pseudo_mutex> g(m, ::std::adopt_lock);
  }
#endif

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// ADDITIONAL_COMPILE_FLAGS: -DTEST_NO_HIP_THREAD

// <mutex>

// template <class Mutex> class lock_guard;

// Make sure that the implicitly-generated CTAD works.

#include <hip/mutex>

#include "force_include_hip.h"

#include "test_macros.h"
#include "types.h"

int main(int, char**) {
#ifdef __HIP_DEVICE_COMPILE__
  MyMutex m;
  {
    hip::lock_guard lg(m);
    ASSERT_SAME_TYPE(decltype(lg), hip::lock_guard<MyMutex>);
  }
#endif

  return 0;
}


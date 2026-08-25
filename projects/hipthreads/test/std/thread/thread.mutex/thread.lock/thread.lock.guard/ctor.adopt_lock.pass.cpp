//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// ADDITIONAL_COMPILE_FLAGS: -DTEST_NO_HIP_THREAD

// <mutex>

// template <class Mutex> class lock_guard;

// lock_guard(mutex_type& m, adopt_lock_t);

#include <mutex>
#include <cassert>
#include <hip/mutex>

#include "force_include_hip.h"

#include "types.h"

int main(int, char**) {
#ifdef __HIP_DEVICE_COMPILE__
  MyMutex m;
  {
    m.lock();
    hip::lock_guard<MyMutex> lg(m, ::std::adopt_lock);
    assert(m.locked);
  }
  assert(!m.locked);
#endif
  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <mutex>

// template <class Mutex> class lock_guard;

// explicit lock_guard(mutex_type& m);

// ADDITIONAL_COMPILE_FLAGS: -DTEST_NO_HIP_THREAD

#include <cassert>
#include <type_traits>
#include <hip/mutex>

#include "force_include_hip.h"

#include "types.h"

int main(int, char**) {
#ifdef __HIP_DEVICE_COMPILE__
  MyMutex m;
  assert(!m.locked);
  hip::lock_guard<MyMutex> lg(m);
  assert(m.locked);
#endif

  static_assert(!::std::is_convertible<MyMutex, hip::lock_guard<MyMutex> >::value, "constructor must be explicit");

  return 0;
}

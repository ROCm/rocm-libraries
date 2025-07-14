//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// ADDITIONAL_COMPILE_FLAGS: -DTEST_USE_GPU_THREADS

// <thread>

// class thread::id

// bool operator==(thread::id x, thread::id y) noexcept;
// bool operator!=(thread::id x, thread::id y) noexcept;
// bool operator< (thread::id x, thread::id y) noexcept;
// bool operator<=(thread::id x, thread::id y) noexcept;
// bool operator> (thread::id x, thread::id y) noexcept;
// bool operator>=(thread::id x, thread::id y) noexcept;
// strong_ordering operator<=>(thread::id x, thread::id y) noexcept;

#include <gpu/thread>
#include <cassert>

#include "test_macros.h"
#include "test_comparisons.h"
#include "force_include_hip.h"

int main(int, char**) {
  AssertComparisonsAreNoexcept<gpu::thread::id>();
  AssertComparisonsReturnBool<gpu::thread::id>();
#if TEST_STD_VER > 17
  AssertOrderAreNoexcept<gpu::thread::id>();
  AssertOrderReturn<std::strong_ordering, gpu::thread::id>();
#endif

  gpu::thread::id id1;
  gpu::thread::id id2;
#ifdef __HIP_DEVICE_COMPILE__
  gpu::thread::id id3 = gpu::this_thread::get_id();
#endif

  // `id1` and `id2` should compare equal
  assert(testComparisons(id1, id2, /*isEqual*/ true, /*isLess*/ false));
#if TEST_STD_VER > 17
  assert(testOrder(id1, id2, std::strong_ordering::equal));
#endif

#ifdef __HIP_DEVICE_COMPILE__
  // Test `t1` and `t3` which are not equal
  bool isLess = id1 < id3;
  assert(testComparisons(id1, id3, /*isEqual*/ false, isLess));
#if TEST_STD_VER > 17
  assert(testOrder(id1, id3, isLess ? std::strong_ordering::less : std::strong_ordering::greater));
#endif
#endif

  // Regression tests for https://github.com/llvm/llvm-project/issues/56187
  // libc++ previously declared the comparison operators as hidden friends
  // which was non-conforming.
  assert(gpu::operator==(id1, id2));
#if TEST_STD_VER <= 17
  assert(!gpu::operator!=(id1, id2));
  assert(!gpu::operator<(id1, id2));
  assert(gpu::operator<=(id1, id2));
  assert(!gpu::operator>(id1, id2));
  assert(gpu::operator>=(id1, id2));
#else
  assert(gpu::operator<=>(id1, id2) == std::strong_ordering::equal);
#endif

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads

// <condition_variable>

// class condition_variable_any;

// void notify_all();

#include <hip/condition_variable>
#include <hip/mutex>
#include <hip/thread>
#include <vector>
#include <cassert>

#include <hip/atomic>

#include "make_test_thread.h"
#include "test_macros.h"

#include "force_include_hip.h"

__device__ hip::condition_variable_any cv;

typedef hip::spin_mutex L0;
typedef hip::unique_lock<L0> L1;

__device__ L0 m0;

__device__  bool pleaseExit = false;
__device__  hip::std::atomic<unsigned> notReady;

__device__ void helper() {
  L1 lk(m0);
  --notReady;
  while (pleaseExit == false)
    cv.wait(lk);
}

int main(int, char**)
{
#ifdef __HIP_DEVICE_COMPILE__
  const unsigned threadCount = 2;
  notReady = threadCount;
  hip::wthread threads[threadCount];
  for (unsigned i = 0; i < threadCount; i++)
    threads[i] = support::make_test_thread(helper);
  {
    while (notReady > 0)
      hip::this_thread::pseudo_yield();
    // At this point, both threads have had a chance to acquire the lock and are
    // either waiting on the condition variable or about to wait.
    L1 lk(m0);
    pleaseExit = true;
    // POSIX does not guarantee reliable scheduling if notify_all is called
    // without the lock being held.
    cv.notify_all();
  }
  // The test will hang if not all of the threads were woken.
  for (unsigned i = 0; i < threadCount; i++)
    threads[i].join();
#endif

  return 0;
}

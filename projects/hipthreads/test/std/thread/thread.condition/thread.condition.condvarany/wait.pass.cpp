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

// template <class Lock>
//   void wait(Lock& lock);

#include <hip/condition_variable>
#include <hip/mutex>
#include <hip/thread>
#include <cassert>

#include "make_test_thread.h"
#include "test_macros.h"

#include "force_include_hip.h"

__device__ hip::condition_variable_any cv;

typedef hip::spin_mutex L0;
typedef hip::unique_lock<L0> L1;

__device__ L0 m0;

__device__ int test1 = 0;
__device__ int test2 = 0;

__device__ void f()
{
    L1 lk(m0);
    assert(test2 == 0);
    test1 = 1;
    cv.notify_one();
    while (test2 == 0)
        cv.wait(lk);
    assert(test2 != 0);
}

int main(int, char**)
{
#ifdef __HIP_DEVICE_COMPILE__
    L1 lk(m0);
    hip::wthread t = support::make_test_thread(f);
    assert(test1 == 0);
    while (test1 == 0)
        cv.wait(lk);
    assert(test1 != 0);
    test2 = 1;
    lk.unlock();
    cv.notify_one();
    t.join();
#endif

  return 0;
}

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

// class condition_variable;

// void wait(unique_lock<mutex>& lock);

#include <hip/condition_variable>
#include <hip/mutex>
#include <hip/thread>
#include <cassert>

#include "make_test_thread.h"
#include "test_macros.h"

#include "force_include_hip.h"

__device__ hip::spin_condition_variable cv;
__device__ hip::spin_mutex mut;

__device__ int test1 = 0;
__device__ int test2 = 0;

__device__ void f()
{
    hip::unique_lock<hip::spin_mutex> lk(mut);
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
    hip::unique_lock<hip::spin_mutex> lk(mut);
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

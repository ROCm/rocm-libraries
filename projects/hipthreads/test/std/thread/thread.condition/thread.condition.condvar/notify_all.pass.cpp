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

// void notify_all();

#include <hip/atomic>
#include <hip/condition_variable>
#include <hip/mutex>
#include <hip/thread>
#include <cassert>

#include "make_test_thread.h"
#include "test_macros.h"

#include "force_include_hip.h"

__device__ hip::spin_condition_variable cv;
__device__ hip::spin_mutex mut;

__device__ int test0 = 0;
__device__ int test1 = 0;
__device__ int test2 = 0;

__device__ hip::std::atomic<int> ready_count(0);

__device__ void f1()
{
    hip::unique_lock<hip::spin_mutex> lk(mut);
    assert(test1 == 0);
    ready_count += 1;
    while (test1 == 0)
        cv.wait(lk);
    assert(test1 == 1);
    test1 = 2;
}

__device__ void f2()
{
    hip::unique_lock<hip::spin_mutex> lk(mut);
    assert(test2 == 0);
    ready_count += 1;
    while (test2 == 0)
        cv.wait(lk);
    assert(test2 == 1);
    test2 = 2;
}

int main(int, char**)
{
#ifdef __HIP_DEVICE_COMPILE__
    hip::wthread t1 = support::make_test_thread(f1);
    hip::wthread t2 = support::make_test_thread(f2);
    while (ready_count.load() != 2) {
      hip::this_thread::sleep_for(cuda::std::chrono::milliseconds(100));
    }
    {
        hip::unique_lock<hip::spin_mutex>lk(mut);
        test1 = 1;
        test2 = 1;
    }
    cv.notify_all();
    {
        hip::this_thread::sleep_for(cuda::std::chrono::milliseconds(100));
        hip::unique_lock<hip::spin_mutex>lk(mut);
    }
    t1.join();
    t2.join();
    assert(test1 == 2);
    assert(test2 == 2);
#endif
  return 0;
}

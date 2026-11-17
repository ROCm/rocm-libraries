//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// ALLOW_RETRIES: 2

// <condition_variable>

// class condition_variable_any;

// void notify_one();

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

__device__ int test0 = 0;
__device__ int test1 = 0;
__device__ int test2 = 0;

__device__ void f1()
{
    L1 lk(m0);
    assert(test1 == 0);
    while (test1 == 0)
        cv.wait(lk);
    assert(test1 == 1);
    test1 = 2;
}

__device__ void f2()
{
    L1 lk(m0);
    assert(test2 == 0);
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
    hip::this_thread::sleep_for(cuda::std::chrono::milliseconds(100));
    {
        L1 lk(m0);
        test1 = 1;
        test2 = 1;
    }
    cv.notify_one();
    {
        hip::this_thread::sleep_for(cuda::std::chrono::milliseconds(100));
        L1 lk(m0);
    }
    if (test1 == 2)
    {
        t1.join();
        test1 = 0;
    }
    else if (test2 == 2)
    {
        t2.join();
        test2 = 0;
    }
    else
        assert(false);
    cv.notify_one();
    {
        hip::this_thread::sleep_for(cuda::std::chrono::milliseconds(100));
        L1 lk(m0);
    }
    if (test1 == 2)
    {
        t1.join();
        test1 = 0;
    }
    else if (test2 == 2)
    {
        t2.join();
        test2 = 0;
    }
    else
        assert(false);
#endif

  return 0;
}

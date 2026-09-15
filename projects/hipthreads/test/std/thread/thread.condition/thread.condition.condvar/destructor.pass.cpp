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

// ~condition_variable();

#include <hip/thread>
#include <cassert>
#include <hip/condition_variable>
#include <hip/mutex>
#include <cstdio>

#include "make_test_thread.h"
#include "test_macros.h"

#include "force_include_hip.h"

__device__ hip::spin_condition_variable* cv;
__device__ hip::spin_mutex m;
typedef hip::unique_lock<hip::spin_mutex> Lock;

__device__ bool f_ready = false;
__device__ bool g_ready = false;

__device__ void f()
{
    Lock lk(m);
    f_ready = true;
    cv->notify_one();
    delete cv;
}

__device__ void g()
{
    Lock lk(m);
    g_ready = true;
    cv->notify_one();
    while (!f_ready)
        cv->wait(lk);
}

int main(int, char**)
{
#ifdef __HIP_DEVICE_COMPILE__
    cv = new hip::spin_condition_variable;
    hip::wthread th2 = support::make_test_thread(g);
    Lock lk(m);
    while (!g_ready)
        cv->wait(lk);
    lk.unlock();
    hip::wthread th1 = support::make_test_thread(f);
    th1.join();
    th2.join();
#endif
  return 0;
}

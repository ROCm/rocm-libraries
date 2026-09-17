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

// ~condition_variable_any();

#include <hip/condition_variable>
#include <hip/mutex>
#include <hip/thread>
#include <cassert>

#include "make_test_thread.h"
#include "test_macros.h"

#include "force_include_hip.h"

__device__ hip::condition_variable_any* cv;
__device__ hip::spin_mutex m;

__device__ bool f_ready = false;
__device__ bool g_ready = false;

__device__ void f()
{
    m.lock();
    f_ready = true;
    cv->notify_one();
    delete cv;
    m.unlock();
}

__device__ void g()
{
    m.lock();
    g_ready = true;
    cv->notify_one();
    while (!f_ready)
        cv->wait(m);
    m.unlock();
}

int main(int, char**)
{
#ifdef __HIP_DEVICE_COMPILE__
    cv = new hip::condition_variable_any;
    hip::wthread th2 = support::make_test_thread(g);
    m.lock();
    while (!g_ready)
        cv->wait(m);
    m.unlock();
    hip::wthread th1 = support::make_test_thread(f);
    th1.join();
    th2.join();
#endif

  return 0;
}

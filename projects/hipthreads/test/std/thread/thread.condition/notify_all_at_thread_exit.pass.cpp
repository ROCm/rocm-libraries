//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// XFAIL: *
// REASON: not yet implemented in hipThreads

// notify_all_at_thread_exit(...) requires move semantics to transfer the unique_lock.
// UNSUPPORTED: c++03

// <condition_variable>
//
// void notify_all_at_thread_exit(condition_variable& cond, unique_lock<mutex> lk);

#include <condition_variable>
#include <hip/thread>
#include <hip/std/chrono>
#include <hip/condition_variable>
#include <hip/mutex>
#include <cassert>

#include "make_test_thread.h"
#include "test_macros.h"

#include "force_include_hip.h"

hip::spin_condition_variable cv;
hip::spin_mutex mut;

typedef cuda::std::chrono::milliseconds ms;
typedef cuda::std::chrono::high_resolution_clock Clock;

__device__ void func()
{
    hip::unique_lock<hip::spin_mutex> lk(mut);
    ::std::notify_all_at_thread_exit(cv, ::std::move(lk));
    hip::this_thread::sleep_for(ms(300));
}

int main(int, char**)
{
#ifdef __HIP_DEVICE_COMPILE__
    hip::unique_lock<hip::spin_mutex> lk(mut);
    hip::wthread t = support::make_test_thread(func);
    Clock::time_point t0 = Clock::now();
    cv.wait(lk);
    Clock::time_point t1 = Clock::now();
    assert(t1-t0 > ms(250));
    t.join();
#endif

  return 0;
}

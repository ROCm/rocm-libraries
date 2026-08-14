//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11

// <shared_mutex>

// class shared_timed_mutex;

// template <class Clock, class Duration>
//   shared_lock(mutex_type& m, const chrono::time_point<Clock, Duration>& abs_time);

#include <hip/thread>

#include <atomic>
#include <hip/std/chrono>
#include <cassert>
#include <cstdlib>
#include <shared_mutex>
#include <vector>

#include "make_test_thread.h"
#include "test_macros.h"

::std::shared_timed_mutex m;

typedef cuda::std::chrono::steady_clock Clock;
typedef Clock::time_point time_point;
typedef Clock::duration duration;
typedef cuda::std::chrono::milliseconds ms;
typedef cuda::std::chrono::nanoseconds ns;

ms LongTime = ms(5000);
ms ShortTime = ms(50);

static constexpr unsigned Threads = 5;

::std::atomic<unsigned> CountDown(Threads);

void f1()
{
  --CountDown;
  time_point t0 = Clock::now();
  ::std::shared_lock<::std::shared_timed_mutex> lk(m, t0 + LongTime);
  time_point t1 = Clock::now();
  assert(lk.owns_lock() == true);
  assert(t1 - t0 <= LongTime);
}

void f2()
{
  time_point t0 = Clock::now();
  ::std::shared_lock<::std::shared_timed_mutex> lk(m, t0 + ShortTime);
  time_point t1 = Clock::now();
  assert(lk.owns_lock() == false);
  assert(t1 - t0 >= ShortTime);
}

int main(int, char**)
{
  {
    m.lock();
    ::std::vector<hip::wthread> v;
    for (unsigned i = 0; i < Threads; ++i)
      v.push_back(support::make_test_thread(f1));
    while (CountDown > 0)
      hip::this_thread::pseudo_yield();
    hip::this_thread::sleep_for(ShortTime);
    m.unlock();
    for (auto& t : v)
      t.join();
  }
  {
    m.lock();
    ::std::vector<hip::wthread> v;
    for (unsigned i = 0; i < Threads; ++i)
      v.push_back(support::make_test_thread(f2));
    for (auto& t : v)
      t.join();
    m.unlock();
  }

  return 0;
}

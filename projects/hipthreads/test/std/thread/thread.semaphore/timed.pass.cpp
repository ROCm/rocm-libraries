//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14, c++17

// XFAIL: availability-synchronization_library-missing

// <semaphore>

#include <semaphore>
#include <hip/thread>
#include <hip/std/chrono>
#include <cassert>

#include "make_test_thread.h"
#include "test_macros.h"

int main(int, char**)
{
  auto const start = cuda::std::chrono::steady_clock::now();

  ::std::counting_semaphore<> s(0);

  assert(!s.try_acquire_until(start + cuda::std::chrono::milliseconds(250)));
  assert(!s.try_acquire_for(cuda::std::chrono::milliseconds(250)));

  hip::wthread t = support::make_test_thread([&](){
    hip::this_thread::sleep_for(cuda::std::chrono::milliseconds(250));
    s.release();
    hip::this_thread::sleep_for(cuda::std::chrono::milliseconds(250));
    s.release();
  });

  assert(s.try_acquire_until(start + cuda::std::chrono::seconds(2)));
  assert(s.try_acquire_for(cuda::std::chrono::seconds(2)));
  t.join();

  auto const end = cuda::std::chrono::steady_clock::now();
  assert(end - start < cuda::std::chrono::seconds(10));

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: no-threads
// REQUIRES: thread-safety
// ADDITIONAL_COMPILE_FLAGS: -D_LIBHIPTHREADS_ENABLE_THREAD_SAFETY_ANNOTATIONS

// On Windows Clang bugs out when both __declspec and __attribute__ are present,
// the processing goes awry preventing the definition of the types.
// XFAIL: msvc

// <shared_mutex>
//
// class shared_timed_mutex;
//
// void lock();
// bool try_lock();
// bool try_lock_for(const cuda::std::chrono::duration<Rep, Period>&);
// bool try_lock_until(const cuda::std::chrono::time_point<Clock, Duration>&);
// void unlock();
//
// void lock_shared();
// bool try_lock_shared();
// bool try_lock_shared_for(const cuda::std::chrono::duration<Rep, Period>&);
// bool try_lock_shared_until(const cuda::std::chrono::time_point<Clock, Duration>&);
// void unlock_shared();

#include <hip/std/chrono>
#include <shared_mutex>

::std::shared_timed_mutex m;
int data __attribute__((guarded_by(m))) = 0;
void read(int);

void f(cuda::std::chrono::time_point<cuda::std::chrono::steady_clock> tp, cuda::std::chrono::milliseconds d) {
  // Exclusive locking
  {
    m.lock();
    ++data; // ok
    m.unlock();
  }
  {
    if (m.try_lock()) {
      ++data; // ok
      m.unlock();
    }
  }
  {
    if (m.try_lock_for(d)) {
      ++data; // ok
      m.unlock();
    }
  }
  {
    if (m.try_lock_until(tp)) {
      ++data; // ok
      m.unlock();
    }
  }

  // Shared locking
  {
    m.lock_shared();
    read(data); // ok
    ++data; // expected-error {{writing variable 'data' requires holding shared_timed_mutex 'm' exclusively}}
    m.unlock_shared();
  }
  {
    if (m.try_lock_shared()) {
      read(data); // ok
      ++data; // expected-error {{writing variable 'data' requires holding shared_timed_mutex 'm' exclusively}}
      m.unlock_shared();
    }
  }
  {
    if (m.try_lock_shared_for(d)) {
      read(data); // ok
      ++data; // expected-error {{writing variable 'data' requires holding shared_timed_mutex 'm' exclusively}}
      m.unlock_shared();
    }
  }
  {
    if (m.try_lock_shared_until(tp)) {
      read(data); // ok
      ++data; // expected-error {{writing variable 'data' requires holding shared_timed_mutex 'm' exclusively}}
      m.unlock_shared();
    }
  }
}

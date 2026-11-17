//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads

// <mutex>

// Make sure hip::unique_lock works with ::std::mutex as expected.

#include <cassert>
#include <mutex>
#include <hip/atomic>
#include <hip/mutex>

#include "force_include_hip.h"

#include "make_test_thread.h"

__device__ hip::std::atomic<bool> keep_waiting;
__device__ hip::std::atomic<bool> child_thread_locked;
__device__ hip::spin_mutex mux;
__device__ bool main_thread_unlocked  = false;
__device__ bool child_thread_unlocked = false;

__device__ void lock_thread() {
  hip::unique_lock<hip::spin_mutex> lock(mux);
  assert(main_thread_unlocked);
  main_thread_unlocked  = false;
  child_thread_unlocked = true;
}

__device__ void try_lock_thread() {
  hip::unique_lock<hip::spin_mutex> lock(mux, ::std::try_to_lock_t());
  assert(lock.owns_lock());
  child_thread_locked = true;

  while (keep_waiting)
    hip::this_thread::sleep_for(cuda::std::chrono::milliseconds(10));

  child_thread_unlocked = true;
}

int main(int, char**) {
#ifdef __HIP_DEVICE_COMPILE__
  {
    mux.lock();
    hip::wthread t        = support::make_test_thread(lock_thread);
    main_thread_unlocked = true;
    mux.unlock();
    t.join();
    assert(child_thread_unlocked);
  }

  {
    child_thread_unlocked = false;
    child_thread_locked   = false;
    keep_waiting          = true;
    hip::wthread t         = support::make_test_thread(try_lock_thread);
    while (!child_thread_locked)
      hip::this_thread::sleep_for(cuda::std::chrono::milliseconds(10));
    assert(!mux.try_lock());
    keep_waiting = false;
    t.join();
    assert(child_thread_unlocked);
  }
#endif

  return 0;
}

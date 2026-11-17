//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_THREAD_THREAD_MUTEX_THREAD_LOCK_THREAD_LOCK_GUARD_TYPES_H
#define TEST_STD_THREAD_THREAD_MUTEX_THREAD_LOCK_THREAD_LOCK_GUARD_TYPES_H

#include <cassert>

struct MyMutex {
  __host__ __device__ bool locked = false;

  __host__ __device__ MyMutex() = default;
  __host__ __device__ ~MyMutex() { assert(!locked); }

  __host__ __device__ void lock() {
    assert(!locked);
    locked = true;
  }
  __host__ __device__ void unlock() {
    assert(locked);
    locked = false;
  }

  MyMutex(MyMutex const&)            = delete;
  MyMutex& operator=(MyMutex const&) = delete;
};

#endif // TEST_STD_THREAD_THREAD_MUTEX_THREAD_LOCK_THREAD_LOCK_GUARD_TYPES_H

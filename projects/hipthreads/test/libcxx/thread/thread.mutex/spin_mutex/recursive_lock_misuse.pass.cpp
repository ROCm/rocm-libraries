//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads

// <mutex>

// hip::spin_mutex / hip::pseudo_mutex are non-recursive. Re-locking from a block
// that already holds the lock is misuse: in a release (NDEBUG) build it used to
// silently livelock, because the guard was a plain assert() that NDEBUG strips.
// The library now routes the misuse through its device misuse path (see
// hip/__support/misuse.h), which aborts via __builtin_trap() in normal use.
//
// In nonfatal mode the misuse is recorded in __hipthreads_misuse_count and the
// offending lock() returns without spinning, so the contract is verifiable here
// without livelocking and without depending on an unstable abort exit code.

#include <hip/mutex>
#include <hip/pseudo_mutex>
#include <cassert>

#include "test_macros.h"

#include "force_include_hip.h"

int main(int, char**) {
#ifdef __HIP_DEVICE_COMPILE__
  hip::__hipthreads_misuse_nonfatal = true;

  // spin_mutex: recursive lock from the same block is detected.
  {
    hip::spin_mutex m;
    m.lock();
    unsigned int before = hip::__hipthreads_misuse_count;
    m.lock();  // misuse: recursive lock -> detected + skipped in nonfatal mode
    assert(hip::__hipthreads_misuse_count == before + 1);
    m.unlock();
  }

  // pseudo_mutex: same contract.
  {
    hip::pseudo_mutex m;
    m.lock();
    unsigned int before = hip::__hipthreads_misuse_count;
    m.lock();  // misuse: recursive lock -> detected + skipped in nonfatal mode
    assert(hip::__hipthreads_misuse_count == before + 1);
    m.unlock();
  }
#endif

  return 0;
}

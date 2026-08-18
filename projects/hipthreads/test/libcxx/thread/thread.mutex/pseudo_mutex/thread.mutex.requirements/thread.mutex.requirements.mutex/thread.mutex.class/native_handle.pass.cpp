//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads, libcpp-has-thread-api-external
// ADDITIONAL_COMPILE_FLAGS: -DTEST_NO_HIP_THREAD

// XFAIL: windows

// pseudo_mutex currently don't support native_handle()
// XFAIL: *

// <mutex>

// class mutex;

// typedef pthread_mutex_t* native_handle_type;
// native_handle_type native_handle();

#include <hip/pseudo_mutex>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    hip::pseudo_mutex m;
    pthread_mutex_t* h = m.native_handle();
    assert(h);

  return 0;
}

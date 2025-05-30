//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads, libcpp-has-thread-api-external

// XFAIL: windows

// spin_condition_variable currently don't support native_handle()
// XFAIL: *

// <condition_variable>

// class condition_variable;

// typedef pthread_cond_t* native_handle_type;
// native_handle_type native_handle();

#include <cassert>
#include <gpu/condition_variable>
#include <pthread.h>
#include <type_traits>

#include "test_macros.h"

int main(int, char**)
{
    static_assert((std::is_same<gpu::spin_condition_variable::native_handle_type,
                                pthread_cond_t*>::value), "");
    gpu::spin_condition_variable cv;
    gpu::spin_condition_variable::native_handle_type h = cv.native_handle();
    assert(h != nullptr);
  return 0;
}

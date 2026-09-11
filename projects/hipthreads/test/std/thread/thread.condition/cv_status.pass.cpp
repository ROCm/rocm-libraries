//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// ADDITIONAL_COMPILE_FLAGS: -DTEST_NO_HIP_THREAD
// XFAIL: *
// REASON: not yet implemented in hipThreads

// <condition_variable>

// enum class cv_status { no_timeout, timeout };

#include <condition_variable>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    assert(static_cast<int>(hip::cv_status::no_timeout) == 0);
    assert(static_cast<int>(hip::cv_status::timeout)    == 1);

  return 0;
}

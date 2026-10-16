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

// <condition_variable>

// class condition_variable_any;

// condition_variable_any();

#include <cassert>

#include "test_macros.h"
#include <hip/condition_variable>
#include "force_include_hip.h"

int main(int, char**)
{
#ifdef __HIP_DEVICE_COMPILE__
    [[maybe_unused]] hip::condition_variable_any cv;
#endif

  return 0;
}

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

// class condition_variable;

// condition_variable();

#include <cassert>

#include "force_include_hip.h"
#include <hip/condition_variable>

int main(int, char**)
{
#ifdef __HIP_DEVICE_COMPILE__
    hip::spin_condition_variable cv;
    static_cast<void>(cv);
#endif

  return 0;
}

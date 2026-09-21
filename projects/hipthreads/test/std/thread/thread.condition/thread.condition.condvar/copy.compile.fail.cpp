//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <condition_variable>

// class condition_variable;

// condition_variable(const condition_variable&) = delete;

// ADDITIONAL_COMPILE_FLAGS: -DTEST_NO_HIP_THREAD

#include <cassert>

#include "force_include_hip.h"
#include <hip/condition_variable>

int main(int, char**)
{
#ifdef __HIP_DEVICE_COMPILE__
    hip::spin_condition_variable cv0;
    hip::spin_condition_variable cv1(cv0);
#endif

  return 0;
}

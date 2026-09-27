//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <condition_variable>

// class condition_variable_any;

// condition_variable_any& operator=(const condition_variable_any&) = delete;

// ADDITIONAL_COMPILE_FLAGS: -DTEST_NO_HIP_THREAD

#include <cassert>

#include <hip/condition_variable>
#include "force_include_hip.h"

int main(int, char**)
{
#ifdef __HIP_DEVICE_COMPILE__
    hip::condition_variable_any cv0;
    hip::condition_variable_any cv1;
    cv1 = cv0;
#endif

  return 0;
}

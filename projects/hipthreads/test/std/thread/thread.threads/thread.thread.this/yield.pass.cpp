//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads

// <thread>

// void this_thread::yield();

#include <hip/thread>
#include <cassert>

#include "force_include_hip.h"

#include "test_macros.h"

int main(int, char**)
{
#ifdef __HIP_DEVICE_COMPILE__
    hip::this_thread::pseudo_yield();
#endif

  return 0;
}

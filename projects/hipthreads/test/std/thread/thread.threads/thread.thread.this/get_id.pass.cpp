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

// wthread::id this_thread::get_id();

#include <hip/thread>
#include <cassert>

#include "test_macros.h"

#include "force_include_hip.h"

int main(int, char**)
{
#ifdef __HIP_DEVICE_COMPILE__
    hip::wthread::id id = hip::this_thread::get_id();
    assert(id != hip::wthread::id());
#endif

  return 0;
}

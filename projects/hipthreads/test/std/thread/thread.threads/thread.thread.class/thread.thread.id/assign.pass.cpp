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

// class wthread::id

// id& operator=(const id&) = default;

#include <hip/thread>
#include <cassert>

#include "test_macros.h"

#include "force_include_hip.h"

int main(int, char**)
{
    hip::wthread::id id0;
    hip::wthread::id id1;
    id1 = id0;
    assert(id1 == id0);
#ifdef __HIP_DEVICE_COMPILE__
    id1 = hip::this_thread::get_id();
    assert(id1 != id0);
#endif

  return 0;
}

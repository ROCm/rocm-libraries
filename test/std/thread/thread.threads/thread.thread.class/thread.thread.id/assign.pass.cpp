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

// class thread::id

// id& operator=(const id&) = default;

#include <gpu/thread>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    gpu::thread::id id0;
    gpu::thread::id id1;
    id1 = id0;
    assert(id1 == id0);
    id1 = gpu::this_thread::get_id();
    assert(id1 != id0);

  return 0;
}

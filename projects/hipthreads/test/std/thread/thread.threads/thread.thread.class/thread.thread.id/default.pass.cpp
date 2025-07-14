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

// id();

#include <gpu/thread>
#include <cassert>

#include "test_macros.h"
#include "force_include_hip.h"

int main(int, char**)
{
    gpu::thread::id id;
    assert(id == gpu::thread::id());

  return 0;
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <thread>

// class wthread

// wthread(const wthread&) = delete;

#include <hip/thread>

#include "force_include_hip.h"

int main(int, char**)
{
    hip::wthread t0; (void)t0;
    hip::wthread t1(t0); (void)t1;
    return 0;
}

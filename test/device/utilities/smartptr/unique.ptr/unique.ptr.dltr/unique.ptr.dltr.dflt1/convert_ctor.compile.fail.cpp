//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// default_delete

// Test that default_delete<T[]> does not have a working converting constructor

#include "gpu/memory"
#include <cassert>

#include "kernel_launcher.h"

struct A
{
};

struct B
    : public A
{
};

__global__ void gmain()
{
    gpu::default_delete<B[]> d2;
    gpu::default_delete<A[]> d1 = d2;
}

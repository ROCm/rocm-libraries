//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gpu/memory"
#include <cassert>

#include "kernel_launcher.h"

__global__ void gmain()
{
    auto up2 = gpu::make_unique<int[]>(10, 20, 30, 40);
}

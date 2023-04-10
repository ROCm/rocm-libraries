//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gpu/memory"
#include <string>
#include <cassert>

int main(int, char**)
{
    // TODO: Update once we implement gpu::string
    auto up1 = gpu::make_unique<gpu::string[]>("error"); // doesn't compile - no bound

  return 0;
}

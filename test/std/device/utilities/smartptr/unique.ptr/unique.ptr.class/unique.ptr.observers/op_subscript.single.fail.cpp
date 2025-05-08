//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// test op[](size_t)

#include "gpu/memory"
#include <cassert>

#include "kernel_launcher.h"

__global__ void gmain() {
  gpu::unique_ptr<int> p(new int[3]);
  gpu::unique_ptr<int> const& cp = p;
  p[0];  // expected-error {{type 'gpu::unique_ptr<int>' does not provide a subscript operator}}
  cp[1]; // expected-error {{type 'const gpu::unique_ptr<int>' does not provide a subscript operator}}

}

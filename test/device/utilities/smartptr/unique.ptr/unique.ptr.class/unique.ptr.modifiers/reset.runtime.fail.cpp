//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>
// UNSUPPORTED: c++03

// unique_ptr

// test reset

#include "gpu/memory"
#include <cassert>

#include "unique_ptr_test_helper.h"
#include "kernel_launcher.h"

__global__ void gmain() {
  {
    gpu::unique_ptr<A[]> p;
    p.reset(static_cast<B*>(nullptr)); // expected-error {{no matching member function for call to 'reset'}}
  }
  {
    gpu::unique_ptr<int[]> p;
    p.reset(static_cast<const int*>(nullptr)); // expected-error {{no matching member function for call to 'reset'}}
  }

}

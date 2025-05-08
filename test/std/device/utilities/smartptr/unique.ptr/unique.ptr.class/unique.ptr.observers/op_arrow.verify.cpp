//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// test op->()

#include "gpu/memory"
#include <cassert>

#include "kernel_launcher.h"

struct V {
  int member;
};

__global__ void gmain() {
  gpu::unique_ptr<V[]> p;
  gpu::unique_ptr<V[]> const& cp = p;

  p->member; // expected-error-re {{member reference type 'gpu::unique_ptr<V{{[ ]*}}[]>' is not a pointer}}
             // expected-error@-1 {{no member named 'member'}}

  cp->member; // expected-error-re {{member reference type 'const gpu::unique_ptr<V{{[ ]*}}[]>' is not a pointer}}
              // expected-error@-1 {{no member named 'member'}}

}

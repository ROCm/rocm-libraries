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

struct V {
  int member;
};

int main(int, char **) {
  gpu::unique_ptr_h<V[]> p;
  gpu::unique_ptr_h<V[]> const& cp = p;

  p->member; // expected-error-re {{member reference type 'gpu::unique_ptr_h<V{{[ ]*}}[]>' (aka 'unique_ptr<V{{[ ]*}}[], host_delete<V{{[ ]*}}[]>>') is not a pointer}}
             // expected-error@-1 {{no member named 'member'}}

  cp->member; // expected-error-re {{member reference type 'const gpu::unique_ptr_h<V{{[ ]*}}[]>' (aka 'const unique_ptr<V{{[ ]*}}[], host_delete<V{{[ ]*}}[]>>') is not a pointer}}
              // expected-error@-1 {{no member named 'member'}}

}

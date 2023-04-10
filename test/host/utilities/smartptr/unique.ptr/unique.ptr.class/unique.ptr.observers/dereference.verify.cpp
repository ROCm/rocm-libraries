//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// test op*()

#include "gpu/memory"
#include <cassert>


int main(int, char **) {
  gpu::unique_ptr_h<int[]> p(new int(3));
  const gpu::unique_ptr_h<int[]>& cp = p;
  TEST_IGNORE_NODISCARD(*p);  // expected-error-re {{indirection requires pointer operand ('gpu::unique_ptr_h<int{{[ ]*}}[]>' (aka 'unique_ptr<int{{[ ]*}}[], host_delete<int{{[ ]*}}[]>>') invalid)}}
  TEST_IGNORE_NODISCARD(*cp); // expected-error-re {{indirection requires pointer operand ('const gpu::unique_ptr_h<int{{[ ]*}}[]>' (aka 'const unique_ptr<int{{[ ]*}}[], host_delete<int{{[ ]*}}[]>>') invalid)}}

  return 0;
}

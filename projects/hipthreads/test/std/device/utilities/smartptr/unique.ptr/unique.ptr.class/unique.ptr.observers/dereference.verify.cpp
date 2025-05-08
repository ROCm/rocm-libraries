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

#include "kernel_launcher.h"

__global__ void gmain() {
  gpu::unique_ptr<int[]> p(new int(3));
  const gpu::unique_ptr<int[]>& cp = p;
  TEST_IGNORE_NODISCARD(*p);  // expected-error-re {{indirection requires pointer operand ('gpu::unique_ptr<int{{[ ]*}}[]>' invalid)}}
  TEST_IGNORE_NODISCARD(*cp); // expected-error-re {{indirection requires pointer operand ('const gpu::unique_ptr<int{{[ ]*}}[]>' invalid)}}

}

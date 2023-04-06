//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This code triggers https://gcc.gnu.org/bugzilla/show_bug.cgi?id=104568
// UNSUPPORTED: msvc

// Test the fix for https://llvm.org/PR54100

#include "gpu/memory"
#include <cassert>

#include "kernel_launcher.h"
#include "test_macros.h"

struct A {
  int m[0];
};
static_assert(sizeof(A) == 0, "");  // an extension supported by GCC and Clang

__global__ void gmain()
{
  {
    gpu::unique_ptr<A> p = gpu::unique_ptr<A>(new A);
    assert(p != nullptr);
  }
  {
    gpu::unique_ptr<A[]> p = gpu::unique_ptr<A[]>(new A[1]);
    assert(p != nullptr);
  }
#if TEST_STD_VER > 11
  {
    gpu::unique_ptr<A> p = gpu::make_unique<A>();
    assert(p != nullptr);
  }
  {
    gpu::unique_ptr<A[]> p = gpu::make_unique<A[]>(1);
    assert(p != nullptr);
  }
#endif
}

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// unique_ptr<T, const D&>(pointer, D()) should not compile

#include "gpu/memory"
#include "kernel_launcher.h"

struct Deleter {
  __device__ void operator()(int* p) const { delete p; }
};

__global__ void gmain() {
  // expected-error@+1 {{call to deleted constructor of 'gpu::unique_ptr<int, const Deleter &>}}
  gpu::unique_ptr<int, const Deleter&> s((int*)nullptr, Deleter());

}

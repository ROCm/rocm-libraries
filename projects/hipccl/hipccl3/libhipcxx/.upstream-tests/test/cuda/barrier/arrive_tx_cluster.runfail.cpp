//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: pre-sm-90

// UNSUPPORTED: no_execute

// NOTE(HIP/AMD): currently barrier is not supported on AMD hardware
// UNSUPPORTED: hipcc, hiprtc

// <cuda/barrier>

#include <cuda/barrier>

#include "test_macros.h"
#include <cooperative_groups.h>

// Suppress warning about barrier in shared memory
TEST_NV_DIAG_SUPPRESS(static_var_with_dynamic_init)

int main(int, char**)
{
  NV_DISPATCH_TARGET(
    NV_IS_HOST,
    (
      // When PR #416 is merged, uncomment this line:
      // cuda_cluster_size = 2;
      ),
    NV_IS_DEVICE,
    (__shared__ cuda::barrier<cuda::thread_scope_block> bar;

     if (threadIdx.x == 0) { init(&bar, blockDim.x); } namespace cg = cooperative_groups;
     auto cluster                                                   = cg::this_cluster();

     cluster.sync();

     // This test currently fails at this point because support for
     // clusters has not yet been added.
     cuda::barrier<cuda::thread_scope_block> * remote_bar;
     remote_bar = cluster.map_shared_rank(&bar, cluster.block_rank() ^ 1);

     // When PR #416 is merged, this should fail here because the barrier
     // is in device memory.
     auto token = cuda::device::barrier_arrive_tx(*remote_bar, 1, 0);));
  return 0;
}

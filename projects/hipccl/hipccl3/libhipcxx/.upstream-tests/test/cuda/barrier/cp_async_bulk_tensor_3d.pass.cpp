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

// UNSUPPORTED: nvrtc, hiprtc
// XFAIL: clang && !nvcc
// NVRTC_SKIP_KERNEL_RUN // This will have effect once PR 433 is merged (line above should be removed.)

// NOTE(HIP/AMD): currently barrier is not supported on AMD hardware
// UNSUPPORTED: hipcc

// <cuda/barrier>

#include <cuda/barrier>
#include <cuda/std/array>

#include "cp_async_bulk_tensor_generic.h"

// Define the size of contiguous tensor in global and shared memory.
//
// Note that the first dimension is the one with stride 1. This one must be a
// multiple of 4 to ensure that each new dimension starts at a 16-byte aligned
// offset.
//
// We have a separate variable for host and device because a constexpr
// cuda::std::array cannot be shared between host and device as some of its
// member functions take a const reference, which is unsupported by nvcc.
constexpr cuda::std::array<uint64_t, 3> GMEM_DIMS{8, 11, 13};
__device__ constexpr cuda::std::array<uint64_t, 3> GMEM_DIMS_DEV{8, 11, 13};
constexpr cuda::std::array<uint32_t, 3> SMEM_DIMS{4, 2, 4};
__device__ constexpr cuda::std::array<uint32_t, 3> SMEM_DIMS_DEV{4, 2, 4};

__device__ constexpr cuda::std::array<uint32_t, 3> TEST_SMEM_COORDS[] = {{0, 0, 0}, {4, 1, 3}, {4, 5, 1}};

constexpr size_t gmem_len = tensor_len(GMEM_DIMS);
constexpr size_t smem_len = tensor_len(SMEM_DIMS);

__device__ int gmem_tensor[gmem_len];

int main(int, char**)
{
  NV_DISPATCH_TARGET(
    NV_IS_HOST,
    (
      // Required by concurrent_agents_launch to know how many we're launching
      cuda_thread_count = 512; init_tensor_map(gmem_tensor, GMEM_DIMS, SMEM_DIMS);),
    NV_IS_DEVICE,
    (for (auto smem_coord : TEST_SMEM_COORDS) {
      test<smem_len>(smem_coord, SMEM_DIMS_DEV, GMEM_DIMS_DEV, gmem_tensor, gmem_len);
    }));
  return 0;
}

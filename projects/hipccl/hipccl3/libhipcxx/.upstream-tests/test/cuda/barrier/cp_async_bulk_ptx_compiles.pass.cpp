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

// NOTE(HIP/AMD): currently barrier is not supported on AMD hardware
// UNSUPPORTED: hipcc, hiprtc

// <cuda/barrier>

#include <cuda/barrier>

#include "test_macros.h" // TEST_NV_DIAG_SUPPRESS

// Suppress warning about barrier in shared memory
TEST_NV_DIAG_SUPPRESS(static_var_with_dynamic_init)

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// Kernels below are intended to be compiled, but not run. This is to check if
// all generated PTX is valid.
__global__ void test_bulk_tensor(CUtensorMap* map)
{
  __shared__ int smem;
#if _CCCL_CUDA_COMPILER(CLANG)
  __shared__ char barrier_data[sizeof(barrier)];
  barrier& bar = cuda::std::bit_cast<barrier>(barrier_data);
#else // ^^^ _CCCL_CUDA_COMPILER(CLANG) ^^^ / vvv !_CCCL_CUDA_COMPILER(CLANG)
  __shared__ barrier bar;
#endif // !_CCCL_CUDA_COMPILER(CLANG)
  if (threadIdx.x == 0)
  {
    init(&bar, blockDim.x);
  }
  __syncthreads();

  cde::cp_async_bulk_tensor_1d_global_to_shared(&smem, map, 0, bar);
  cde::cp_async_bulk_tensor_2d_global_to_shared(&smem, map, 0, 0, bar);
  cde::cp_async_bulk_tensor_3d_global_to_shared(&smem, map, 0, 0, 0, bar);
  cde::cp_async_bulk_tensor_4d_global_to_shared(&smem, map, 0, 0, 0, 0, bar);
  cde::cp_async_bulk_tensor_5d_global_to_shared(&smem, map, 0, 0, 0, 0, 0, bar);

  cde::cp_async_bulk_tensor_1d_shared_to_global(map, 0, &smem);
  cde::cp_async_bulk_tensor_2d_shared_to_global(map, 0, 0, &smem);
  cde::cp_async_bulk_tensor_3d_shared_to_global(map, 0, 0, 0, &smem);
  cde::cp_async_bulk_tensor_4d_shared_to_global(map, 0, 0, 0, 0, &smem);
  cde::cp_async_bulk_tensor_5d_shared_to_global(map, 0, 0, 0, 0, 0, &smem);
}

__global__ void test_bulk(void* gmem)
{
  __shared__ int smem;
  __shared__ char barrier_data[sizeof(barrier)];
  barrier& bar = *reinterpret_cast<barrier*>(&barrier_data);
  if (threadIdx.x == 0)
  {
    init(&bar, blockDim.x);
  }
  __syncthreads();

  cde::cp_async_bulk_global_to_shared(&smem, gmem, 1024, bar);
  cde::cp_async_bulk_shared_to_global(gmem, &smem, 1024);
}

__global__ void test_fences_async_group(void*)
{
  cde::fence_proxy_async_shared_cta();

  cde::cp_async_bulk_commit_group();
  // Wait for up to 8 groups
  cde::cp_async_bulk_wait_group_read<0>();
  cde::cp_async_bulk_wait_group_read<1>();
  cde::cp_async_bulk_wait_group_read<2>();
  cde::cp_async_bulk_wait_group_read<3>();
  cde::cp_async_bulk_wait_group_read<4>();
  cde::cp_async_bulk_wait_group_read<5>();
  cde::cp_async_bulk_wait_group_read<6>();
  cde::cp_async_bulk_wait_group_read<7>();
  cde::cp_async_bulk_wait_group_read<8>();
}

int main(int, char**)
{
  return 0;
}

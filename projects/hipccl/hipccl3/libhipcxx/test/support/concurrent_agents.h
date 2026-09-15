//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// MIT License
//
// Modifications Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef _CONCURRENT_AGENTS_H
#define _CONCURRENT_AGENTS_H

#if !defined(__CUDA_ARCH__) && !defined(__HIPCC_RTC__)
#  include <thread>
#endif

#include <cuda/std/cassert>

#include "test_macros.h"

_CCCL_EXEC_CHECK_DISABLE
template <class Fun>
__host__ __device__ void execute_on_main_thread(Fun&& fun)
{
  NV_IF_ELSE_TARGET(NV_IS_DEVICE, (if (threadIdx.x == 0) { fun(); } __syncthreads();), (fun();))
}

template <typename... Fs>
__host__ __device__ void concurrent_agents_launch(Fs... fs)
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (assert(blockDim.x == sizeof...(Fs)); using fptr = void (*)(void*);

     fptr device_threads[] = {[](void* data) {
       (*reinterpret_cast<Fs*>(data))();
     }...};

     void* device_thread_data[] = {reinterpret_cast<void*>(&fs)...};

     __syncthreads();

     device_threads[threadIdx.x](device_thread_data[threadIdx.x]);

     __syncthreads();),
    (std::thread threads[]{std::thread{std::forward<Fs>(fs)}...};

     for (auto&& thread : threads) { thread.join(); }))
}

#endif // _CONCURRENT_AGENTS_H

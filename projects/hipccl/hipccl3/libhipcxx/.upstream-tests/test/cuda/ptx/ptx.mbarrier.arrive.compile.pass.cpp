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

// NOTE(HIP/AMD): AMD does not have a PTX equivalent
// UNSUPPORTED: hipcc, hiprtc

// UNSUPPORTED: libcpp-has-no-threads

// <cuda/ptx>

#include <cuda/ptx>
#include <cuda/std/utility>

/*
 * We use a special strategy to force the generation of the PTX. This is mainly
 * a fight against dead-code-elimination in the NVVM layer.
 *
 * The reason we need this strategy is because certain older versions of ptxas
 * segfault when a non-sensical sequence of PTX is generated. So instead, we try
 * to force the instantiation and compilation to PTX of all the overloads of the
 * PTX wrapping functions.
 *
 * We do this by writing a function pointer of each overload to the kernel
 * parameter `fn_ptr`.
 *
 * Because `fn_ptr` is possibly visible outside this translation unit, the
 * compiler must compile all the functions which are stored.
 *
 */

__global__ void test_mbarrier_arrive(void** fn_ptr)
{
#if __cccl_ptx_isa >= 700
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // mbarrier.arrive.shared.b64                                  state,  [addr];           // 1.
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<uint64_t (*)(uint64_t*)>(cuda::ptx::mbarrier_arrive));));
#endif // __cccl_ptx_isa >= 700

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (
                   // mbarrier.arrive.shared::cta.b64                             state,  [addr], count;    // 2.
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<uint64_t (*)(uint64_t*, const uint32_t&)>(cuda::ptx::mbarrier_arrive));));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.arrive.release.cta.shared::cta.b64                   state,  [addr];           // 3a.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<uint64_t (*)(
            cuda::ptx::sem_release_t, cuda::ptx::scope_cta_t, cuda::ptx::space_shared_t, uint64_t*)>(
            cuda::ptx::mbarrier_arrive));
          // mbarrier.arrive.release.cluster.shared::cta.b64                   state,  [addr];           // 3a.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint64_t (*)(
                cuda::ptx::sem_release_t, cuda::ptx::scope_cluster_t, cuda::ptx::space_shared_t, uint64_t*)>(
                cuda::ptx::mbarrier_arrive));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.arrive.release.cta.shared::cta.b64                   state,  [addr], count;    // 3b.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<uint64_t (*)(
            cuda::ptx::sem_release_t, cuda::ptx::scope_cta_t, cuda::ptx::space_shared_t, uint64_t*, const uint32_t&)>(
            cuda::ptx::mbarrier_arrive));
          // mbarrier.arrive.release.cluster.shared::cta.b64                   state,  [addr], count;    // 3b.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint64_t (*)(cuda::ptx::sem_release_t,
                                       cuda::ptx::scope_cluster_t,
                                       cuda::ptx::space_shared_t,
                                       uint64_t*,
                                       const uint32_t&)>(cuda::ptx::mbarrier_arrive));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.arrive.release.cluster.shared::cluster.b64                   _, [addr];                // 4a.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::sem_release_t, cuda::ptx::scope_cluster_t, cuda::ptx::space_cluster_t, uint64_t*)>(
            cuda::ptx::mbarrier_arrive));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.arrive.release.cluster.shared::cluster.b64                   _, [addr], count;         // 4b.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::sem_release_t, cuda::ptx::scope_cluster_t, cuda::ptx::space_cluster_t, uint64_t*, const uint32_t&)>(
            cuda::ptx::mbarrier_arrive));));
#endif // __cccl_ptx_isa >= 800
}

__global__ void test_mbarrier_arrive_no_complete(void** fn_ptr)
{
#if __cccl_ptx_isa >= 700
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // mbarrier.arrive.noComplete.shared.b64                       state,  [addr], count;    // 5.
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<uint64_t (*)(uint64_t*, const uint32_t&)>(cuda::ptx::mbarrier_arrive_no_complete));));
#endif // __cccl_ptx_isa >= 700
}

__global__ void test_mbarrier_arrive_expect_tx(void** fn_ptr)
{
#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 state, [addr], tx_count; // 8.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<uint64_t (*)(
            cuda::ptx::sem_release_t, cuda::ptx::scope_cta_t, cuda::ptx::space_shared_t, uint64_t*, const uint32_t&)>(
            cuda::ptx::mbarrier_arrive_expect_tx));
          // mbarrier.arrive.expect_tx.release.cluster.shared::cta.b64 state, [addr], tx_count; // 8.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<uint64_t (*)(cuda::ptx::sem_release_t,
                                       cuda::ptx::scope_cluster_t,
                                       cuda::ptx::space_shared_t,
                                       uint64_t*,
                                       const uint32_t&)>(cuda::ptx::mbarrier_arrive_expect_tx));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64   _, [addr], tx_count; // 9.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::sem_release_t, cuda::ptx::scope_cluster_t, cuda::ptx::space_cluster_t, uint64_t*, const uint32_t&)>(
            cuda::ptx::mbarrier_arrive_expect_tx));));
#endif // __cccl_ptx_isa >= 800
}

int main(int, char**)
{
  return 0;
}

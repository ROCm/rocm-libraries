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

__global__ void test_mbarrier_test_wait(void** fn_ptr)
{
#if __cccl_ptx_isa >= 700
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // mbarrier.test_wait.shared.b64 waitComplete, [addr], state; // 1.
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<bool (*)(uint64_t*, const uint64_t&)>(cuda::ptx::mbarrier_test_wait));));
#endif // __cccl_ptx_isa >= 700

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.test_wait.acquire.cta.shared::cta.b64        waitComplete, [addr], state; // 2.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<bool (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, uint64_t*, const uint64_t&)>(
            cuda::ptx::mbarrier_test_wait));
          // mbarrier.test_wait.acquire.cluster.shared::cta.b64        waitComplete, [addr], state; // 2.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<bool (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, uint64_t*, const uint64_t&)>(
                cuda::ptx::mbarrier_test_wait));));
#endif // __cccl_ptx_isa >= 800
}

__global__ void test_mbarrier_test_wait_parity(void** fn_ptr)
{
#if __cccl_ptx_isa >= 710
  NV_IF_TARGET(NV_PROVIDES_SM_80,
               (
                   // mbarrier.test_wait.parity.shared.b64 waitComplete, [addr], phaseParity; // 3.
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<bool (*)(uint64_t*, const uint32_t&)>(cuda::ptx::mbarrier_test_wait_parity));));
#endif // __cccl_ptx_isa >= 710

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 waitComplete, [addr], phaseParity; // 4.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<bool (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, uint64_t*, const uint32_t&)>(
            cuda::ptx::mbarrier_test_wait_parity));
          // mbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 waitComplete, [addr], phaseParity; // 4.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<bool (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, uint64_t*, const uint32_t&)>(
                cuda::ptx::mbarrier_test_wait_parity));));
#endif // __cccl_ptx_isa >= 800
}

__global__ void test_mbarrier_try_wait(void** fn_ptr)
{
#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (
                   // mbarrier.try_wait.shared::cta.b64         waitComplete, [addr], state; // 5a.
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<bool (*)(uint64_t*, const uint64_t&)>(cuda::ptx::mbarrier_try_wait));));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.try_wait.shared::cta.b64         waitComplete, [addr], state, suspendTimeHint;                    //
        // 5b.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<bool (*)(uint64_t*, const uint64_t&, const uint32_t&)>(cuda::ptx::mbarrier_try_wait));));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.try_wait.acquire.cta.shared::cta.b64         waitComplete, [addr], state;                        //
        // 6a.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<bool (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, uint64_t*, const uint64_t&)>(
            cuda::ptx::mbarrier_try_wait));
          // mbarrier.try_wait.acquire.cluster.shared::cta.b64         waitComplete, [addr], state; // 6a.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<bool (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, uint64_t*, const uint64_t&)>(
                cuda::ptx::mbarrier_try_wait));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.try_wait.acquire.cta.shared::cta.b64         waitComplete, [addr], state , suspendTimeHint;      //
        // 6b.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<bool (*)(
            cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, uint64_t*, const uint64_t&, const uint32_t&)>(
            cuda::ptx::mbarrier_try_wait));
          // mbarrier.try_wait.acquire.cluster.shared::cta.b64         waitComplete, [addr], state , suspendTimeHint;
          // // 6b.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<bool (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, uint64_t*, const uint64_t&, const uint32_t&)>(
                cuda::ptx::mbarrier_try_wait));));
#endif // __cccl_ptx_isa >= 800
}

__global__ void test_mbarrier_try_wait_parity(void** fn_ptr)
{
#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (
                   // mbarrier.try_wait.parity.shared::cta.b64  waitComplete, [addr], phaseParity; // 7a.
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<bool (*)(uint64_t*, const uint32_t&)>(cuda::ptx::mbarrier_try_wait_parity));));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.try_wait.parity.shared::cta.b64  waitComplete, [addr], phaseParity, suspendTimeHint; // 7b.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<bool (*)(uint64_t*, const uint32_t&, const uint32_t&)>(cuda::ptx::mbarrier_try_wait_parity));));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.try_wait.parity.acquire.cta.shared::cta.b64  waitComplete, [addr], phaseParity;                  //
        // 8a.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<bool (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, uint64_t*, const uint32_t&)>(
            cuda::ptx::mbarrier_try_wait_parity));
          // mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64  waitComplete, [addr], phaseParity; // 8a.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<bool (*)(cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, uint64_t*, const uint32_t&)>(
                cuda::ptx::mbarrier_try_wait_parity));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // mbarrier.try_wait.parity.acquire.cta.shared::cta.b64  waitComplete, [addr], phaseParity, suspendTimeHint; //
        // 8b.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<bool (*)(
            cuda::ptx::sem_acquire_t, cuda::ptx::scope_cta_t, uint64_t*, const uint32_t&, const uint32_t&)>(
            cuda::ptx::mbarrier_try_wait_parity));
          // mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64  waitComplete, [addr], phaseParity,
          // suspendTimeHint; // 8b.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<bool (*)(
                cuda::ptx::sem_acquire_t, cuda::ptx::scope_cluster_t, uint64_t*, const uint32_t&, const uint32_t&)>(
                cuda::ptx::mbarrier_try_wait_parity));));
#endif // __cccl_ptx_isa >= 800
}

int main(int, char**)
{
  return 0;
}

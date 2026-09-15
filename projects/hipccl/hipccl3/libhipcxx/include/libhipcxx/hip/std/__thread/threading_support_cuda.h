// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

#ifndef _LIBCUDACXX___THREAD_THREADING_SUPPORT_CUDA_H
#define _LIBCUDACXX___THREAD_THREADING_SUPPORT_CUDA_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !defined(_LIBCUDACXX_HAS_NO_THREADS) && defined(_LIBCUDACXX_HAS_THREAD_API_CUDA)

#  include <cuda/std/chrono>
#  include <cuda/std/climits>

_CCCL_PUSH_MACROS

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_LIBCUDACXX_HIDE_FROM_ABI void __cccl_thread_yield() {}

_LIBCUDACXX_HIDE_FROM_ABI void __cccl_thread_sleep_for(_CUDA_VSTD::chrono::nanoseconds __ns)
{
  #ifdef __HIP_DEVICE_COMPILE__
// FIXME(HIP): Due to a lack of a clock rate-independent nanosleep ISA instruction on AMD GPUs,
// we use wall_clock64() that runs at a fixed clock rate to simulate a nanosleep (only on AMD GPUs).
// The clock-rate depends on the device architecture (e.g., 25 MHz on gfx90a, 100 MHz on gfx942).
// Therefore, clock rate is currently hardcoded as _LIBCUDACXX_HIP_TSC_NANOSECONDS_PER_CYCLE. 
// We will need to find a way to make this information available either at compile time or query it at runtime.
// The default way to query hipDeviceAttributeWallClockRate does not work on NVIDIA devices (ROCm 5.4, tested on A100).
// The default way to query hipDeviceAttributeWallClockRate() does not work on NVIDIA devices (ROCm 5.4, tested on A100).
NV_IF_TARGET_LIBHIPCXX(
  NV_IS_DEVICE_LIBHIPCXX, (
      auto const __step = __ns.count();
      long long  __now, __start;
      __now = wall_clock64();
      __start = __now; 

#ifdef _LIBCUDACXX_HIP_USE_FINEGRANULAR_NANOSLEEP        
      while((__now-__start)*_LIBCUDACXX_HIP_TSC_NANOSECONDS_PER_CYCLE<__step) {
          __now = wall_clock64();
      }
#else
      // This is derived from https://github.com/RadeonOpenCompute/ROCm-Device-Libs (rtcwait).
      // Fixme(HIP): Make this architecture-aware
      long long __end = __now + __step/_LIBCUDACXX_HIP_TSC_NANOSECONDS_PER_CYCLE;

      while (__end > __now + 1625) {
          __builtin_amdgcn_s_sleep(127);
          __now = wall_clock64();
      }

      while (__end > __now + 806) {
          __builtin_amdgcn_s_sleep(63);
          __now = wall_clock64();
      }

      while (__end > __now + 396) {
          __builtin_amdgcn_s_sleep(31);
          __now = wall_clock64();
      }

      while (__end > __now + 192) {
          __builtin_amdgcn_s_sleep(15);
          __now = wall_clock64();
      }

      while (__end > __now + 89) {
          __builtin_amdgcn_s_sleep(7);
          __now = wall_clock64();
      }

      while (__end > __now + 38) {
          __builtin_amdgcn_s_sleep(3);
          __now = wall_clock64();
      }

      while (__end > __now) {
          __builtin_amdgcn_s_sleep(1);
          __now = wall_clock64();
      }
#endif
    )
  )
#else
  NV_IF_TARGET_LIBHIPCXX(NV_IS_DEVICE_LIBHIPCXX,
               (auto const __step = __ns.count(); assert(__step < numeric_limits<unsigned>::max());
                asm volatile("nanosleep.u32 %0;" ::"r"((unsigned) __step) :);))
#endif
}

_LIBCUDACXX_END_NAMESPACE_STD

_CCCL_POP_MACROS

#endif // !_LIBCUDACXX_HAS_NO_THREADS && _LIBCUDACXX_HAS_THREAD_API_CUDA

#endif // _LIBCUDACXX___THREAD_THREADING_SUPPORT_CUDA_H

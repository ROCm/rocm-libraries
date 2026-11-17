// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

#ifndef __LIBHIPTHREADS___THREAD_THIS_THREAD_H__
#define __LIBHIPTHREADS___THREAD_THIS_THREAD_H__

/**
 * @file
 * @brief Utilities operating on the calling (current) GPU thread context.
 * @ingroup thread
 *
 * Facilities mirror a subset of `std::this_thread` adapted for HIP GPU
 * execution plus CUDA/HIP–specific helpers:
 *
 *  - sleep_for(duration): Busy / pseudo blocking wait for at least the
 *    specified duration (device/host variants routed through scheduler).
 *  - pseudo_yield(): Cooperative hint allowing other GPU work to progress.
 *  - get_width(): Returns logical participation width for the current
 *    work node (number of active lanes in this wthread).
 *  - get_fiber_id(): Internal / diagnostic identifier for the current
 *    fiber / lane (stable only within the lifetime of the work node).
 *
 * Notes:
 *  - Durations use cuda::std::chrono types (distinct from ::std).
 *  - pseudo_yield() allows the scheduler to switch to other ready work, but
 *    provides no guarantees about which wthread runs next, when the yielding
 *    wthread resumes, or scheduling fairness. Additionally, the yielding wthread
 *    cannot resume until the yieldee completes, which can cause deadlock in
 *    yield-loop scenarios (see pseudo_mutex documentation).
 *  - Width may be < warp size when a wthread was launched with an
 *    explicit width parameter.
 */

#include <hip/std/chrono>

namespace cuda {

namespace this_thread {

/**
 * @brief Sleep (busy / scheduler mediated) for at least @p __ns.
 * @param __ns Duration in nanoseconds (cuda::std::chrono).
 *
 * May spin or cooperatively yield internally; precision not guaranteed.
 */
_LIBHIPTHREADS_EXPORTED_FROM_ABI __host__ __device__ void sleep_for(cuda::std::chrono::nanoseconds __ns);

// TODO: Should we also provide an implementation that accepts ::std::chrono::duration (and not just cuda::std::chrono::duration)?
/**
 * @brief Templated convenience overload forwarding to nanosecond sleep.
 * @tparam _Rep Rep type.
 * @tparam _Period Period ratio.
 * @param __d Duration to sleep (negative or zero -> no-op).
 *
 * Rounds up to the next nanosecond if needed; clamps at nanoseconds::max().
 */
template <class _Rep, class _Period>
__host__ __device__ _LIBHIPTHREADS_HIDE_FROM_ABI void sleep_for(const cuda::std::chrono::duration<_Rep, _Period>& __d) {
  if (__d > cuda::std::chrono::duration<_Rep, _Period>::zero()) {
    // The standard guarantees a 64bit signed integer resolution for nanoseconds,
    // so use INT64_MAX / 1e9 as cut-off point. Use a constant to avoid <climits>
    // and issues with long double folding on PowerPC with GCC.
    constexpr cuda::std::chrono::duration<long double> __max{9223372036.0L};
    cuda::std::chrono::nanoseconds __ns;
    if (__d < __max) {
      __ns = cuda::std::chrono::duration_cast<cuda::std::chrono::nanoseconds>(__d);
      if (__ns < __d)
        ++__ns;
    } else
      __ns = cuda::std::chrono::nanoseconds::max();
    hip::this_thread::sleep_for(__ns);
  }
}

/**
 * @brief Cooperative yield hint.
 *
 * Allows scheduler / runtime to switch to other ready work. No ordering
 * or fairness guarantee.
 */
__device__ void pseudo_yield();

/**
 * @brief Returns logical width (active lanes) of current work node.
 * @return Number of lanes participating (>=1).
 */
__device__ unsigned int get_width() noexcept;

/**
 * @brief Returns the current fiber's unique identifier within this thread.
 * @return Zero-based fiber index, stable within the lifetime of the work unit.
 *
 * Commonly used for loop indexing and work distribution across fibers.
 * Example: `for (uint32_t i = get_fiber_id(); i < n; i += get_width())`
 */
__device__ unsigned int get_fiber_id() noexcept;

} // namespace this_thread

} // namespace cuda

#endif // __LIBHIPTHREADS___THREAD_THIS_THREAD_H__

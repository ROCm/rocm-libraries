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

#ifndef __GPU___THREAD_THIS_THREAD_H__
#define __GPU___THREAD_THIS_THREAD_H__

#include <hip/std/chrono>

namespace gpu {

namespace this_thread {

_LIBGPU_EXPORTED_FROM_ABI __device__ void sleep_for(cuda::std::chrono::nanoseconds __ns);

// TODO: Should we also provide an implementation that accepts std::chrono::duration (and not just cuda::std::chrono::duration)?
template <class _Rep, class _Period>
__device__ _LIBGPU_HIDE_FROM_ABI void sleep_for(const cuda::std::chrono::duration<_Rep, _Period>& __d) {
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
    gpu::this_thread::sleep_for(__ns);
  }
}

__device__ gpu::thread::id get_id() noexcept;
__device__ void pseudo_yield();
__device__ unsigned int get_width() noexcept;
__device__ unsigned int get_fiber_id() noexcept;

} // namespace this_thread

} // namespace gpu

#endif // __GPU___THREAD_THIS_THREAD_H__

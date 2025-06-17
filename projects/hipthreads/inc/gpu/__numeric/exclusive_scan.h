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

#ifndef __GPU___NUMERIC_EXCLUSIVE_SCAN_H
#define __GPU___NUMERIC_EXCLUSIVE_SCAN_H

#include "gpu/__config"

namespace gpu {

#if _LIBGPU_STD_VER >= 17

template <class _InputIterator, class _OutputIterator, class _Tp, class _BinaryOp>
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20 _OutputIterator
exclusive_scan(_InputIterator __first, _InputIterator __last, _OutputIterator __result, _Tp __init, _BinaryOp __b) {
  if (__first != __last) {
    _Tp __tmp(__b(__init, *__first));
    while (true) {
      *__result = std::move(__init);
      ++__result;
      ++__first;
      if (__first == __last)
        break;
      __init = std::move(__tmp);
      __tmp = __b(__init, *__first);
    }
  }
  return __result;
}

template <class _InputIterator, class _OutputIterator, class _Tp>
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20 _OutputIterator
exclusive_scan(_InputIterator __first, _InputIterator __last, _OutputIterator __result, _Tp __init) {
  return gpu::exclusive_scan(__first, __last, __result, __init, std::plus<>());
}

#endif // _LIBGPU_STD_VER >= 17

} // namespace gpu

#endif // __GPU___NUMERIC_EXCLUSIVE_SCAN_H

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

#ifndef __GPU___NUMERIC_TRANSFORM_INCLUSIVE_SCAN_H
#define __GPU___NUMERIC_TRANSFORM_INCLUSIVE_SCAN_H

#include "gpu/__config"

namespace gpu {

#if _LIBGPU_STD_VER >= 17

template <class _InputIterator, class _OutputIterator, class _Tp, class _BinaryOp, class _UnaryOp>
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
_OutputIterator
transform_inclusive_scan(_InputIterator __first, _InputIterator __last,
                           _OutputIterator __result, _BinaryOp __b, _UnaryOp __u, _Tp __init)
{
    for (; __first != __last; ++__first, (void) ++__result) {
        __init = __b(__init, __u(*__first));
        *__result = __init;
        }

    return __result;
}

template <class _InputIterator, class _OutputIterator, class _BinaryOp, class _UnaryOp>
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
_OutputIterator
transform_inclusive_scan(_InputIterator __first, _InputIterator __last,
                               _OutputIterator __result, _BinaryOp __b, _UnaryOp __u)
{
    if (__first != __last) {
        typename std::iterator_traits<_InputIterator>::value_type __init = __u(*__first);
        *__result++ = __init;
        if (++__first != __last)
            return gpu::transform_inclusive_scan(__first, __last, __result, __b, __u, __init);
    }

    return __result;
}

#endif // _LIBGPU_STD_VER >= 17

} // namespace gpu

#endif // __GPU___NUMERIC_TRANSFORM_INCLUSIVE_SCAN_H

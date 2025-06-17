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

#ifndef __GPU___ALGORITHM_SHIFT_LEFT_H
#define __GPU___ALGORITHM_SHIFT_LEFT_H

#include "gpu/__config"

namespace gpu {

#if _LIBGPU_STD_VER >= 20

template <class _ForwardIterator>
__device__ inline _LIBGPU_INLINE_VISIBILITY constexpr
_ForwardIterator
shift_left(_ForwardIterator __first, _ForwardIterator __last,
           typename std::iterator_traits<_ForwardIterator>::difference_type __n)
{
    if (__n == 0) {
        return __last;
    }

    _ForwardIterator __m = __first;
    if constexpr (__is_cpp17_random_access_iterator<_ForwardIterator>::value) {
        if (__n >= __last - __first) {
            return __first;
        }
        __m += __n;
    } else {
        for (; __n > 0; --__n) {
            if (__m == __last) {
                return __first;
            }
            ++__m;
        }
    }
    return gpu::move(__m, __last, __first);
}

#endif // _LIBGPU_STD_VER >= 20

} // namespace gpu

#endif // __GPU___ALGORITHM_SHIFT_LEFT_H

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

#ifndef __GPU___ALGORITHM_PARTITION_POINT_H
#define __GPU___ALGORITHM_PARTITION_POINT_H

#include "gpu/__config"

namespace gpu {

template<class _ForwardIterator, class _Predicate>
__device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 _ForwardIterator
partition_point(_ForwardIterator __first, _ForwardIterator __last, _Predicate __pred)
{
    typedef typename std::iterator_traits<_ForwardIterator>::difference_type difference_type;
    difference_type __len = gpu::distance(__first, __last);
    while (__len != 0)
    {
        difference_type __l2 = std::__half_positive(__len);
        _ForwardIterator __m = __first;
        gpu::advance(__m, __l2);
        if (__pred(*__m))
        {
            __first = ++__m;
            __len -= __l2 + 1;
        }
        else
            __len = __l2;
    }
    return __first;
}

} // namespace gpu

#endif // __GPU___ALGORITHM_PARTITION_POINT_H

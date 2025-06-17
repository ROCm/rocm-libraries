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

#ifndef __GPU___ALGORITHM_IS_HEAP_UNTIL_H
#define __GPU___ALGORITHM_IS_HEAP_UNTIL_H

#include "gpu/__config"

namespace gpu {

template <class _Compare, class _RandomAccessIterator>
__device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 _RandomAccessIterator
__is_heap_until(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare&& __comp)
{
    typedef typename std::iterator_traits<_RandomAccessIterator>::difference_type difference_type;
    difference_type __len = __last - __first;
    difference_type __p = 0;
    difference_type __c = 1;
    _RandomAccessIterator __pp = __first;
    while (__c < __len)
    {
        _RandomAccessIterator __cp = __first + __c;
        if (__comp(*__pp, *__cp))
            return __cp;
        ++__c;
        ++__cp;
        if (__c == __len)
            return __last;
        if (__comp(*__pp, *__cp))
            return __cp;
        ++__p;
        ++__pp;
        __c = 2 * __p + 1;
    }
    return __last;
}

template <class _RandomAccessIterator, class _Compare>
_LIBGPU_NODISCARD_EXT __device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 _RandomAccessIterator
is_heap_until(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp)
{
    return gpu::__is_heap_until(__first, __last, static_cast<__comp_ref_type<_Compare> >(__comp));
}

template<class _RandomAccessIterator>
_LIBGPU_NODISCARD_EXT __device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 _RandomAccessIterator
is_heap_until(_RandomAccessIterator __first, _RandomAccessIterator __last)
{
    return gpu::__is_heap_until(__first, __last, __less<typename std::iterator_traits<_RandomAccessIterator>::value_type>());
}

} // namespace gpu

#endif // __GPU___ALGORITHM_IS_HEAP_UNTIL_H

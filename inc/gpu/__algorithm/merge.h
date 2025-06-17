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

#ifndef __GPU___ALGORITHM_MERGE_H
#define __GPU___ALGORITHM_MERGE_H

#include "gpu/__config"

namespace gpu {

template <class _Compare, class _InputIterator1, class _InputIterator2, class _OutputIterator>
__device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20
_OutputIterator
__merge(_InputIterator1 __first1, _InputIterator1 __last1,
        _InputIterator2 __first2, _InputIterator2 __last2, _OutputIterator __result, _Compare __comp)
{
    for (; __first1 != __last1; ++__result)
    {
        if (__first2 == __last2)
            return gpu::copy(__first1, __last1, __result);
        if (__comp(*__first2, *__first1))
        {
            *__result = *__first2;
            ++__first2;
        }
        else
        {
            *__result = *__first1;
            ++__first1;
        }
    }
    return gpu::copy(__first2, __last2, __result);
}

template <class _InputIterator1, class _InputIterator2, class _OutputIterator, class _Compare>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
_OutputIterator
merge(_InputIterator1 __first1, _InputIterator1 __last1,
      _InputIterator2 __first2, _InputIterator2 __last2, _OutputIterator __result, _Compare __comp)
{
    return gpu::__merge<__comp_ref_type<_Compare> >(__first1, __last1, __first2, __last2, __result, __comp);
}

template <class _InputIterator1, class _InputIterator2, class _OutputIterator>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
_OutputIterator
merge(_InputIterator1 __first1, _InputIterator1 __last1,
      _InputIterator2 __first2, _InputIterator2 __last2, _OutputIterator __result)
{
    typedef typename std::iterator_traits<_InputIterator1>::value_type __v1;
    typedef typename std::iterator_traits<_InputIterator2>::value_type __v2;
    return gpu::merge(__first1, __last1, __first2, __last2, __result, __less<__v1, __v2>());
}

} // namespace gpu

#endif // __GPU___ALGORITHM_MERGE_H

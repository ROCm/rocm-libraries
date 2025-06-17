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

#ifndef __GPU___ALGORITHM_LEXICOGRAPHICAL_COMPARE_H
#define __GPU___ALGORITHM_LEXICOGRAPHICAL_COMPARE_H

#include "gpu/__config"

#include <type_traits>
#include <iterator>

#include "gpu/__algorithm/comp.h"
#include "gpu/__algorithm/comp_ref_type.h"

namespace gpu {

template <class _Compare, class _InputIterator1, class _InputIterator2>
__device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 bool
__lexicographical_compare(_InputIterator1 __first1, _InputIterator1 __last1,
                          _InputIterator2 __first2, _InputIterator2 __last2, _Compare __comp)
{
    for (; __first2 != __last2; ++__first1, (void) ++__first2)
    {
        if (__first1 == __last1 || __comp(*__first1, *__first2))
            return true;
        if (__comp(*__first2, *__first1))
            return false;
    }
    return false;
}

template <class _InputIterator1, class _InputIterator2, class _Compare>
_LIBGPU_NODISCARD_EXT __device__ inline
_LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
bool
lexicographical_compare(_InputIterator1 __first1, _InputIterator1 __last1,
                        _InputIterator2 __first2, _InputIterator2 __last2, _Compare __comp)
{
    return gpu::__lexicographical_compare<__comp_ref_type<_Compare> >(__first1, __last1, __first2, __last2, __comp);
}

template <class _InputIterator1, class _InputIterator2>
_LIBGPU_NODISCARD_EXT __device__ inline
_LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
bool
lexicographical_compare(_InputIterator1 __first1, _InputIterator1 __last1,
                        _InputIterator2 __first2, _InputIterator2 __last2)
{
    return gpu::lexicographical_compare(__first1, __last1, __first2, __last2,
                                         __less<typename std::iterator_traits<_InputIterator1>::value_type,
                                                typename std::iterator_traits<_InputIterator2>::value_type>());
}

} // namespace gpu

#endif // __GPU___ALGORITHM_LEXICOGRAPHICAL_COMPARE_H

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

#ifndef __GPU___NUMERIC_PARTIAL_SUM_H
#define __GPU___NUMERIC_PARTIAL_SUM_H

#include "gpu/__config"

namespace gpu {

template <class _InputIterator, class _OutputIterator>
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
_OutputIterator
partial_sum(_InputIterator __first, _InputIterator __last, _OutputIterator __result)
{
    if (__first != __last)
    {
        typename std::iterator_traits<_InputIterator>::value_type __t(*__first);
        *__result = __t;
        for (++__first, (void) ++__result; __first != __last; ++__first, (void) ++__result)
        {
#if _LIBGPU_STD_VER >= 20
            __t = std::move(__t) + *__first;
#else
            __t = __t + *__first;
#endif
            *__result = __t;
        }
    }
    return __result;
}

template <class _InputIterator, class _OutputIterator, class _BinaryOperation>
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
_OutputIterator
partial_sum(_InputIterator __first, _InputIterator __last, _OutputIterator __result,
              _BinaryOperation __binary_op)
{
    if (__first != __last)
    {
        typename std::iterator_traits<_InputIterator>::value_type __t(*__first);
        *__result = __t;
        for (++__first, (void) ++__result; __first != __last; ++__first, (void) ++__result)
        {
#if _LIBGPU_STD_VER >= 20
            __t = __binary_op(std::move(__t), *__first);
#else
            __t = __binary_op(__t, *__first);
#endif
            *__result = __t;
        }
    }
    return __result;
}

} // namespace gpu

#endif // __GPU___NUMERIC_PARTIAL_SUM_H

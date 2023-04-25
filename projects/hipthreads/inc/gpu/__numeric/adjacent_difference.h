// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___NUMERIC_ADJACENT_DIFFERENCE_H
#define __GPU___NUMERIC_ADJACENT_DIFFERENCE_H

#include "gpu/__config"

namespace gpu {

template <class _InputIterator, class _OutputIterator>
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
_OutputIterator
adjacent_difference(_InputIterator __first, _InputIterator __last, _OutputIterator __result)
{
    if (__first != __last)
    {
        typename std::iterator_traits<_InputIterator>::value_type __acc(*__first);
        *__result = __acc;
        for (++__first, (void) ++__result; __first != __last; ++__first, (void) ++__result)
        {
            typename std::iterator_traits<_InputIterator>::value_type __val(*__first);
#if _LIBGPU_STD_VER >= 20
            *__result = __val - std::move(__acc);
#else
            *__result = __val - __acc;
#endif
            __acc = std::move(__val);
        }
    }
    return __result;
}

template <class _InputIterator, class _OutputIterator, class _BinaryOperation>
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
_OutputIterator
adjacent_difference(_InputIterator __first, _InputIterator __last, _OutputIterator __result,
                      _BinaryOperation __binary_op)
{
    if (__first != __last)
    {
        typename std::iterator_traits<_InputIterator>::value_type __acc(*__first);
        *__result = __acc;
        for (++__first, (void) ++__result; __first != __last; ++__first, (void) ++__result)
        {
            typename std::iterator_traits<_InputIterator>::value_type __val(*__first);
#if _LIBGPU_STD_VER >= 20
            *__result = __binary_op(__val, std::move(__acc));
#else
            *__result = __binary_op(__val, __acc);
#endif
            __acc = std::move(__val);
        }
    }
    return __result;
}

} // namespace gpu

#endif // __GPU___NUMERIC_ADJACENT_DIFFERENCE_H

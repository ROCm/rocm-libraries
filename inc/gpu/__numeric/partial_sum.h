// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

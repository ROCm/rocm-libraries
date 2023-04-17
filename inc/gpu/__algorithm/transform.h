//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_TRANSFORM_H
#define __GPU___ALGORITHM_TRANSFORM_H

#include "gpu/__config"

namespace gpu {

template <class _InputIterator, class _OutputIterator, class _UnaryOperation>
inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
_OutputIterator
transform(_InputIterator __first, _InputIterator __last, _OutputIterator __result, _UnaryOperation __op)
{
    for (; __first != __last; ++__first, (void) ++__result)
        *__result = __op(*__first);
    return __result;
}

template <class _InputIterator1, class _InputIterator2, class _OutputIterator, class _BinaryOperation>
inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
_OutputIterator
transform(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2,
          _OutputIterator __result, _BinaryOperation __binary_op)
{
    for (; __first1 != __last1; ++__first1, (void) ++__first2, ++__result)
        *__result = __binary_op(*__first1, *__first2);
    return __result;
}

} // namespace gpu

#endif // __GPU___ALGORITHM_TRANSFORM_H

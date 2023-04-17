//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_COPY_IF_H
#define __GPU___ALGORITHM_COPY_IF_H

#include "gpu/__config"

namespace gpu {

template<class _InputIterator, class _OutputIterator, class _Predicate>
inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
_OutputIterator
copy_if(_InputIterator __first, _InputIterator __last,
        _OutputIterator __result, _Predicate __pred)
{
    for (; __first != __last; ++__first)
    {
        if (__pred(*__first))
        {
            *__result = *__first;
            ++__result;
        }
    }
    return __result;
}

} // namespace gpu

#endif // __GPU___ALGORITHM_COPY_IF_H

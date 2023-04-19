//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_REPLACE_COPY_H
#define __GPU___ALGORITHM_REPLACE_COPY_H

#include "gpu/__config"

namespace gpu {

template <class _InputIterator, class _OutputIterator, class _Tp>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
_OutputIterator
replace_copy(_InputIterator __first, _InputIterator __last, _OutputIterator __result,
             const _Tp& __old_value, const _Tp& __new_value)
{
    for (; __first != __last; ++__first, (void) ++__result)
        if (*__first == __old_value)
            *__result = __new_value;
        else
            *__result = *__first;
    return __result;
}

} // namespace gpu

#endif // __GPU___ALGORITHM_REPLACE_COPY_H

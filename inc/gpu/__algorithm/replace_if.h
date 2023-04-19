//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_REPLACE_IF_H
#define __GPU___ALGORITHM_REPLACE_IF_H

#include "gpu/__config"

namespace gpu {

template <class _ForwardIterator, class _Predicate, class _Tp>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
void
replace_if(_ForwardIterator __first, _ForwardIterator __last, _Predicate __pred, const _Tp& __new_value)
{
    for (; __first != __last; ++__first)
        if (__pred(*__first))
            *__first = __new_value;
}

} // namespace gpu

#endif // __GPU___ALGORITHM_REPLACE_IF_H

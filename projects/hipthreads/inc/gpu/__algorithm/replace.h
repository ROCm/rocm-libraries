//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_REPLACE_H
#define __GPU___ALGORITHM_REPLACE_H

#include "gpu/__config"

namespace gpu {

template <class _ForwardIterator, class _Tp>
inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
void
replace(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __old_value, const _Tp& __new_value)
{
    for (; __first != __last; ++__first)
        if (*__first == __old_value)
            *__first = __new_value;
}

} // namespace gpu

#endif // __GPU___ALGORITHM_REPLACE_H

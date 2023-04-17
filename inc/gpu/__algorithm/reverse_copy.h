//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_REVERSE_COPY_H
#define __GPU___ALGORITHM_REVERSE_COPY_H

#include "gpu/__config"

namespace gpu {

template <class _BidirectionalIterator, class _OutputIterator>
inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
_OutputIterator
reverse_copy(_BidirectionalIterator __first, _BidirectionalIterator __last, _OutputIterator __result)
{
    for (; __first != __last; ++__result)
        *__result = *--__last;
    return __result;
}

} // namespace gpu

#endif // __GPU___ALGORITHM_REVERSE_COPY_H

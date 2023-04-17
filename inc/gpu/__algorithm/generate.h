//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_GENERATE_H
#define __GPU___ALGORITHM_GENERATE_H

#include "gpu/__config"

namespace gpu {

template <class _ForwardIterator, class _Generator>
inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
void
generate(_ForwardIterator __first, _ForwardIterator __last, _Generator __gen)
{
    for (; __first != __last; ++__first)
        *__first = __gen();
}

} // namespace gpu

#endif // __GPU___ALGORITHM_GENERATE_H

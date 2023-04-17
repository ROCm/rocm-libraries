//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_REMOVE_H
#define __GPU___ALGORITHM_REMOVE_H

#include "gpu/__config"

namespace gpu {

template <class _ForwardIterator, class _Tp>
_LIBGPU_NODISCARD_EXT _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 _ForwardIterator
remove(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __value)
{
    __first = gpu::find(__first, __last, __value);
    if (__first != __last)
    {
        _ForwardIterator __i = __first;
        while (++__i != __last)
        {
            if (!(*__i == __value))
            {
                *__first = std::move(*__i);
                ++__first;
            }
        }
    }
    return __first;
}

} // namespace gpu

#endif // __GPU___ALGORITHM_REMOVE_H

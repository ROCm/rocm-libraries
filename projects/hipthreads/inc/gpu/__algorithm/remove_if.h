//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_REMOVE_IF_H
#define __GPU___ALGORITHM_REMOVE_IF_H

#include "gpu/__config"

namespace gpu {

template <class _ForwardIterator, class _Predicate>
_LIBGPU_NODISCARD_EXT _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 _ForwardIterator
remove_if(_ForwardIterator __first, _ForwardIterator __last, _Predicate __pred)
{
    __first = gpu::find_if<_ForwardIterator, _Predicate&>(__first, __last, __pred);
    if (__first != __last)
    {
        _ForwardIterator __i = __first;
        while (++__i != __last)
        {
            if (!__pred(*__i))
            {
                *__first = std::move(*__i);
                ++__first;
            }
        }
    }
    return __first;
}

} // namespace gpu

#endif // __GPU___ALGORITHM_REMOVE_IF_H

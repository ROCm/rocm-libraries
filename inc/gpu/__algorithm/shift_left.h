//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_SHIFT_LEFT_H
#define __GPU___ALGORITHM_SHIFT_LEFT_H

#include "gpu/__config"

namespace gpu {

#if _LIBGPU_STD_VER >= 20

template <class _ForwardIterator>
__device__ inline _LIBGPU_INLINE_VISIBILITY constexpr
_ForwardIterator
shift_left(_ForwardIterator __first, _ForwardIterator __last,
           typename std::iterator_traits<_ForwardIterator>::difference_type __n)
{
    if (__n == 0) {
        return __last;
    }

    _ForwardIterator __m = __first;
    if constexpr (__is_cpp17_random_access_iterator<_ForwardIterator>::value) {
        if (__n >= __last - __first) {
            return __first;
        }
        __m += __n;
    } else {
        for (; __n > 0; --__n) {
            if (__m == __last) {
                return __first;
            }
            ++__m;
        }
    }
    return std::move(__m, __last, __first);
}

#endif // _LIBGPU_STD_VER >= 20

} // namespace gpu

#endif // __GPU___ALGORITHM_SHIFT_LEFT_H

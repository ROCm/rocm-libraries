//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_PARTITION_POINT_H
#define __GPU___ALGORITHM_PARTITION_POINT_H

#include "gpu/__config"

namespace gpu {

template<class _ForwardIterator, class _Predicate>
_LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 _ForwardIterator
partition_point(_ForwardIterator __first, _ForwardIterator __last, _Predicate __pred)
{
    typedef typename std::iterator_traits<_ForwardIterator>::difference_type difference_type;
    difference_type __len = std::distance(__first, __last);
    while (__len != 0)
    {
        difference_type __l2 = std::__half_positive(__len);
        _ForwardIterator __m = __first;
        std::advance(__m, __l2);
        if (__pred(*__m))
        {
            __first = ++__m;
            __len -= __l2 + 1;
        }
        else
            __len = __l2;
    }
    return __first;
}

} // namespace gpu

#endif // __GPU___ALGORITHM_PARTITION_POINT_H

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_IS_SORTED_UNTIL_H
#define __GPU___ALGORITHM_IS_SORTED_UNTIL_H

#include "gpu/__config"

namespace gpu {

template <class _Compare, class _ForwardIterator>
_LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 _ForwardIterator
__is_sorted_until(_ForwardIterator __first, _ForwardIterator __last, _Compare __comp)
{
    if (__first != __last)
    {
        _ForwardIterator __i = __first;
        while (++__i != __last)
        {
            if (__comp(*__i, *__first))
                return __i;
            __first = __i;
        }
    }
    return __last;
}

template <class _ForwardIterator, class _Compare>
_LIBGPU_NODISCARD_EXT inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 _ForwardIterator
is_sorted_until(_ForwardIterator __first, _ForwardIterator __last, _Compare __comp)
{
    return gpu::__is_sorted_until<__comp_ref_type<_Compare> >(__first, __last, __comp);
}

template<class _ForwardIterator>
_LIBGPU_NODISCARD_EXT inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 _ForwardIterator
is_sorted_until(_ForwardIterator __first, _ForwardIterator __last)
{
    return gpu::is_sorted_until(__first, __last, __less<typename std::iterator_traits<_ForwardIterator>::value_type>());
}

} // namespace gpu

#endif // __GPU___ALGORITHM_IS_SORTED_UNTIL_H

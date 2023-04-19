//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_MAX_ELEMENT_H
#define __GPU___ALGORITHM_MAX_ELEMENT_H

#include "gpu/__config"
#include "gpu/__iterator/iterator_traits.h"

namespace gpu {

template <class _Compare, class _ForwardIterator>
__device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14 _ForwardIterator
__max_element(_ForwardIterator __first, _ForwardIterator __last, _Compare __comp)
{
    static_assert(__is_cpp17_forward_iterator<_ForwardIterator>::value,
        "gpu::max_element requires a ForwardIterator");
    if (__first != __last)
    {
        _ForwardIterator __i = __first;
        while (++__i != __last)
            if (__comp(*__first, *__i))
                __first = __i;
    }
    return __first;
}

template <class _ForwardIterator, class _Compare>
_LIBGPU_NODISCARD_EXT __device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14 _ForwardIterator
max_element(_ForwardIterator __first, _ForwardIterator __last, _Compare __comp)
{
    return gpu::__max_element<__comp_ref_type<_Compare> >(__first, __last, __comp);
}


template <class _ForwardIterator>
_LIBGPU_NODISCARD_EXT __device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14 _ForwardIterator
max_element(_ForwardIterator __first, _ForwardIterator __last)
{
    return gpu::max_element(__first, __last,
              __less<typename std::iterator_traits<_ForwardIterator>::value_type>());
}

} // namespace gpu

#endif // __GPU___ALGORITHM_MAX_ELEMENT_H

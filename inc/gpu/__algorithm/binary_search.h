//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_BINARY_SEARCH_H
#define __GPU___ALGORITHM_BINARY_SEARCH_H

#include "gpu/__config"

namespace gpu {

template <class _ForwardIterator, class _Tp, class _Compare>
_LIBGPU_NODISCARD_EXT __device__ inline
_LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
bool
binary_search(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __value, _Compare __comp)
{
    __first = gpu::lower_bound<_ForwardIterator, _Tp, __comp_ref_type<_Compare> >(__first, __last, __value, __comp);
    return __first != __last && !__comp(__value, *__first);
}

template <class _ForwardIterator, class _Tp>
_LIBGPU_NODISCARD_EXT __device__ inline
_LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
bool
binary_search(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __value)
{
    return gpu::binary_search(__first, __last, __value,
                              __less<typename std::iterator_traits<_ForwardIterator>::value_type, _Tp>());
}

} // namespace gpu

#endif // __GPU___ALGORITHM_BINARY_SEARCH_H

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_FILL_H
#define __GPU___ALGORITHM_FILL_H

#include "gpu/__config"

namespace gpu {

// fill isn't specialized for std::memset, because the compiler already optimizes the loop to a call to std::memset.

template <class _ForwardIterator, class _Tp>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
void
__fill(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __value, std::forward_iterator_tag)
{
    for (; __first != __last; ++__first)
        *__first = __value;
}

template <class _RandomAccessIterator, class _Tp>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
void
__fill(_RandomAccessIterator __first, _RandomAccessIterator __last, const _Tp& __value, std::random_access_iterator_tag)
{
    gpu::fill_n(__first, __last - __first, __value);
}

template <class _ForwardIterator, class _Tp>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
void
fill(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __value)
{
    gpu::__fill(__first, __last, __value, typename std::iterator_traits<_ForwardIterator>::iterator_category());
}

} // namespace gpu

#endif // __GPU___ALGORITHM_FILL_H

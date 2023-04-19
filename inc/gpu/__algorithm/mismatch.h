// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_MISMATCH_H
#define __GPU___ALGORITHM_MISMATCH_H

#include "gpu/__config"

namespace gpu {

template <class _InputIterator1, class _InputIterator2, class _BinaryPredicate>
_LIBGPU_NODISCARD_EXT __device__ inline _LIBGPU_INLINE_VISIBILITY
    _LIBGPU_CONSTEXPR_SINCE_CXX20 gpu::pair<_InputIterator1, _InputIterator2>
    mismatch(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2, _BinaryPredicate __pred) {
  for (; __first1 != __last1; ++__first1, (void)++__first2)
    if (!__pred(*__first1, *__first2))
      break;
  return gpu::pair<_InputIterator1, _InputIterator2>(__first1, __first2);
}

template <class _InputIterator1, class _InputIterator2>
_LIBGPU_NODISCARD_EXT __device__ inline _LIBGPU_INLINE_VISIBILITY
    _LIBGPU_CONSTEXPR_SINCE_CXX20 gpu::pair<_InputIterator1, _InputIterator2>
    mismatch(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2) {
  return gpu::mismatch(__first1, __last1, __first2, __equal_to());
}

#if _LIBGPU_STD_VER >= 14
template <class _InputIterator1, class _InputIterator2, class _BinaryPredicate>
_LIBGPU_NODISCARD_EXT __device__ inline _LIBGPU_INLINE_VISIBILITY
    _LIBGPU_CONSTEXPR_SINCE_CXX20 gpu::pair<_InputIterator1, _InputIterator2>
    mismatch(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2, _InputIterator2 __last2,
             _BinaryPredicate __pred) {
  for (; __first1 != __last1 && __first2 != __last2; ++__first1, (void)++__first2)
    if (!__pred(*__first1, *__first2))
      break;
  return gpu::pair<_InputIterator1, _InputIterator2>(__first1, __first2);
}

template <class _InputIterator1, class _InputIterator2>
_LIBGPU_NODISCARD_EXT __device__ inline _LIBGPU_INLINE_VISIBILITY
    _LIBGPU_CONSTEXPR_SINCE_CXX20 gpu::pair<_InputIterator1, _InputIterator2>
    mismatch(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2, _InputIterator2 __last2) {
  return gpu::mismatch(__first1, __last1, __first2, __last2, __equal_to());
}
#endif

} // namespace gpu

#endif // __GPU___ALGORITHM_MISMATCH_H

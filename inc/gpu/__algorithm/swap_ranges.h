//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_SWAP_RANGES_H
#define __GPU___ALGORITHM_SWAP_RANGES_H

#include "gpu/__config"
#include "gpu/__algorithm/iterator_operations.h"
#include "gpu/__algorithm/move_backward.h"
#include "gpu/__utility/pair.h"

namespace gpu {

// 2+2 iterators: the shorter size will be used.
template <class _AlgPolicy, class _ForwardIterator1, class _Sentinel1, class _ForwardIterator2, class _Sentinel2>
_LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20
gpu::pair<_ForwardIterator1, _ForwardIterator2>
__swap_ranges(_ForwardIterator1 __first1, _Sentinel1 __last1, _ForwardIterator2 __first2, _Sentinel2 __last2) {
  while (__first1 != __last1 && __first2 != __last2) {
    _IterOps<_AlgPolicy>::iter_swap(__first1, __first2);
    ++__first1;
    ++__first2;
  }

  return gpu::pair<_ForwardIterator1, _ForwardIterator2>(std::move(__first1), std::move(__first2));
}

// 2+1 iterators: size2 >= size1.
template <class _AlgPolicy, class _ForwardIterator1, class _Sentinel1, class _ForwardIterator2>
_LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20
gpu::pair<_ForwardIterator1, _ForwardIterator2>
__swap_ranges(_ForwardIterator1 __first1, _Sentinel1 __last1, _ForwardIterator2 __first2) {
  while (__first1 != __last1) {
    _IterOps<_AlgPolicy>::iter_swap(__first1, __first2);
    ++__first1;
    ++__first2;
  }

  return gpu::pair<_ForwardIterator1, _ForwardIterator2>(std::move(__first1), std::move(__first2));
}

template <class _ForwardIterator1, class _ForwardIterator2>
inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20 _ForwardIterator2
swap_ranges(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2) {
  return gpu::__swap_ranges<_ClassicAlgPolicy>(
      std::move(__first1), std::move(__last1), std::move(__first2)).second;
}

} // namespace gpu

#endif // __GPU___ALGORITHM_SWAP_RANGES_H

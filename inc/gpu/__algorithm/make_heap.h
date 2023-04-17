//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_MAKE_HEAP_H
#define __GPU___ALGORITHM_MAKE_HEAP_H

#include "gpu/__config"

namespace gpu {

template <class _AlgPolicy, class _Compare, class _RandomAccessIterator>
inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14
void __make_heap(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare&& __comp) {
  __comp_ref_type<_Compare> __comp_ref = __comp;

  using difference_type = typename std::iterator_traits<_RandomAccessIterator>::difference_type;
  difference_type __n = __last - __first;
  if (__n > 1) {
    // start from the first parent, there is no need to consider children
    for (difference_type __start = (__n - 2) / 2; __start >= 0; --__start) {
        std::__sift_down<_AlgPolicy>(__first, __comp_ref, __n, __first + __start);
    }
  }
}

template <class _RandomAccessIterator, class _Compare>
inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20
void make_heap(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp) {
  gpu::__make_heap<_ClassicAlgPolicy>(std::move(__first), std::move(__last), __comp);
}

template <class _RandomAccessIterator>
inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20
void make_heap(_RandomAccessIterator __first, _RandomAccessIterator __last) {
  gpu::make_heap(std::move(__first), std::move(__last),
      __less<typename std::iterator_traits<_RandomAccessIterator>::value_type>());
}

} // namespace gpu

#endif // __GPU___ALGORITHM_MAKE_HEAP_H

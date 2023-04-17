// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_FIND_FIRST_OF_H
#define __GPU___ALGORITHM_FIND_FIRST_OF_H

#include "gpu/__config"

namespace gpu {

template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
_LIBGPU_HIDE_FROM_ABI
_LIBGPU_CONSTEXPR_SINCE_CXX14 _ForwardIterator1 __find_first_of_ce(_ForwardIterator1 __first1,
                                                                   _ForwardIterator1 __last1,
                                                                   _ForwardIterator2 __first2,
                                                                   _ForwardIterator2 __last2,
                                                                   _BinaryPredicate&& __pred) {
  for (; __first1 != __last1; ++__first1)
    for (_ForwardIterator2 __j = __first2; __j != __last2; ++__j)
      if (__pred(*__first1, *__j))
        return __first1;
  return __last1;
}

template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
_LIBGPU_NODISCARD_EXT inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20 _ForwardIterator1
find_first_of(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2,
              _ForwardIterator2 __last2, _BinaryPredicate __pred) {
  return std::__find_first_of_ce(__first1, __last1, __first2, __last2, __pred);
}

template <class _ForwardIterator1, class _ForwardIterator2>
_LIBGPU_NODISCARD_EXT inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20 _ForwardIterator1 find_first_of(
    _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2) {
  return std::__find_first_of_ce(__first1, __last1, __first2, __last2, __equal_to());
}

} // namespace gpu

#endif // __GPU___ALGORITHM_FIND_FIRST_OF_H

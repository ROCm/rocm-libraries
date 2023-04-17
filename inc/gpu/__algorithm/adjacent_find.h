// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_ADJACENT_FIND_H
#define __GPU___ALGORITHM_ADJACENT_FIND_H

#include "gpu/__config"

namespace gpu {

template <class _Iter, class _Sent, class _BinaryPredicate>
_LIBGPU_NODISCARD_EXT _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 _Iter
__adjacent_find(_Iter __first, _Sent __last, _BinaryPredicate&& __pred) {
  if (__first == __last)
    return __first;
  _Iter __i = __first;
  while (++__i != __last) {
    if (__pred(*__first, *__i))
      return __first;
    __first = __i;
  }
  return __i;
}

template <class _ForwardIterator, class _BinaryPredicate>
_LIBGPU_NODISCARD_EXT inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 _ForwardIterator
adjacent_find(_ForwardIterator __first, _ForwardIterator __last, _BinaryPredicate __pred) {
  return gpu::__adjacent_find(std::move(__first), std::move(__last), __pred);
}

template <class _ForwardIterator>
_LIBGPU_NODISCARD_EXT inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 _ForwardIterator
adjacent_find(_ForwardIterator __first, _ForwardIterator __last) {
  return gpu::adjacent_find(std::move(__first), std::move(__last), __equal_to());
}

} // namespace gpu

#endif // __GPU___ALGORITHM_ADJACENT_FIND_H

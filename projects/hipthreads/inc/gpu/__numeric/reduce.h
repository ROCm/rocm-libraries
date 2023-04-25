// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___NUMERIC_REDUCE_H
#define __GPU___NUMERIC_REDUCE_H

#include "gpu/__config"

namespace gpu {

#if _LIBGPU_STD_VER >= 17
template <class _InputIterator, class _Tp, class _BinaryOp>
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20 _Tp reduce(_InputIterator __first, _InputIterator __last,
                                                                   _Tp __init, _BinaryOp __b) {
  for (; __first != __last; ++__first)
    __init = __b(__init, *__first);
  return __init;
}

template <class _InputIterator, class _Tp>
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20 _Tp reduce(_InputIterator __first, _InputIterator __last,
                                                                   _Tp __init) {
  return std::reduce(__first, __last, __init, std::plus<>());
}

template <class _InputIterator>
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20 typename std::iterator_traits<_InputIterator>::value_type
reduce(_InputIterator __first, _InputIterator __last) {
  return std::reduce(__first, __last, typename std::iterator_traits<_InputIterator>::value_type{});
}
#endif

} // namespace gpu

#endif // __GPU___NUMERIC_REDUCE_H

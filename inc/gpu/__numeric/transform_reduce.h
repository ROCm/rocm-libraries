// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___NUMERIC_TRANSFORM_REDUCE_H
#define __GPU___NUMERIC_TRANSFORM_REDUCE_H

#include "gpu/__config"

namespace gpu {

#if _LIBGPU_STD_VER >= 17
template <class _InputIterator, class _Tp, class _BinaryOp, class _UnaryOp>
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20 _Tp transform_reduce(_InputIterator __first,
                                                                             _InputIterator __last, _Tp __init,
                                                                             _BinaryOp __b, _UnaryOp __u) {
  for (; __first != __last; ++__first)
    __init = __b(__init, __u(*__first));
  return __init;
}

template <class _InputIterator1, class _InputIterator2, class _Tp, class _BinaryOp1, class _BinaryOp2>
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20 _Tp transform_reduce(_InputIterator1 __first1,
                                                                             _InputIterator1 __last1,
                                                                             _InputIterator2 __first2, _Tp __init,
                                                                             _BinaryOp1 __b1, _BinaryOp2 __b2) {
  for (; __first1 != __last1; ++__first1, (void)++__first2)
    __init = __b1(__init, __b2(*__first1, *__first2));
  return __init;
}

template <class _InputIterator1, class _InputIterator2, class _Tp>
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20 _Tp transform_reduce(_InputIterator1 __first1,
                                                                             _InputIterator1 __last1,
                                                                             _InputIterator2 __first2, _Tp __init) {
  return std::transform_reduce(__first1, __last1, __first2, std::move(__init), std::plus<>(),
                                 std::multiplies<>());
}
#endif

} // namespace gpu

#endif // __GPU___NUMERIC_TRANSFORM_REDUCE_H

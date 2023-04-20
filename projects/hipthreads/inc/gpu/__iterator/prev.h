// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ITERATOR_PREV_H
#define __GPU___ITERATOR_PREV_H

#include "gpu/__config"
#include "gpu/__iterator/iterator_traits.h"

namespace gpu {

template <class _InputIter>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
    typename std::enable_if<__is_cpp17_input_iterator<_InputIter>::value, _InputIter>::type
    prev(_InputIter __x, typename std::iterator_traits<_InputIter>::difference_type __n = 1) {
  _LIBGPU_ASSERT(__n <= 0 || __is_cpp17_bidirectional_iterator<_InputIter>::value,
                 "Attempt to prev(it, n) with a positive n on a non-bidirectional iterator");
  gpu::advance(__x, -__n);
  return __x;
}

#if _LIBGPU_STD_VER >= 20

// [range.iter.op.prev]

namespace ranges {
namespace __prev {

struct __fn {
  template <bidirectional_iterator _Ip>
  __device__ _LIBGPU_HIDE_FROM_ABI
  constexpr _Ip operator()(_Ip __x) const {
    --__x;
    return __x;
  }

  template <bidirectional_iterator _Ip>
  __device__ _LIBGPU_HIDE_FROM_ABI
  constexpr _Ip operator()(_Ip __x, iter_difference_t<_Ip> __n) const {
    ranges::advance(__x, -__n);
    return __x;
  }

  template <bidirectional_iterator _Ip>
  __device__ _LIBGPU_HIDE_FROM_ABI constexpr _Ip operator()(_Ip __x, iter_difference_t<_Ip> __n, _Ip __bound_iter) const {
    ranges::advance(__x, -__n, __bound_iter);
    return __x;
  }
};

} // namespace __prev

inline namespace __cpo {
  inline constexpr auto prev = __prev::__fn{};
} // namespace __cpo
} // namespace ranges

#endif // _LIBGPU_STD_VER >= 20

} // namespace gpu

#endif // __GPU___ITERATOR_PREV_H

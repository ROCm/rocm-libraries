// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ITERATOR_DISTANCE_H
#define __GPU___ITERATOR_DISTANCE_H

#include "gpu/__config"

namespace gpu {

template <class _InputIter>
inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
typename std::iterator_traits<_InputIter>::difference_type
__distance(_InputIter __first, _InputIter __last, std::input_iterator_tag)
{
    typename std::iterator_traits<_InputIter>::difference_type __r(0);
    for (; __first != __last; ++__first)
        ++__r;
    return __r;
}

template <class _RandIter>
inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
typename std::iterator_traits<_RandIter>::difference_type
__distance(_RandIter __first, _RandIter __last, std::random_access_iterator_tag)
{
    return __last - __first;
}

template <class _InputIter>
inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
typename std::iterator_traits<_InputIter>::difference_type
distance(_InputIter __first, _InputIter __last)
{
    return std::__distance(__first, __last, typename std::iterator_traits<_InputIter>::iterator_category());
}

#if _LIBGPU_STD_VER >= 20

// [range.iter.op.distance]

namespace ranges {
namespace __distance {

struct __fn {
  template<class _Ip, sentinel_for<_Ip> _Sp>
    requires (!sized_sentinel_for<_Sp, _Ip>)
  _LIBGPU_HIDE_FROM_ABI
  constexpr iter_difference_t<_Ip> operator()(_Ip __first, _Sp __last) const {
    iter_difference_t<_Ip> __n = 0;
    while (__first != __last) {
      ++__first;
      ++__n;
    }
    return __n;
  }

  template<class _Ip, sized_sentinel_for<std::decay_t<_Ip>> _Sp>
  _LIBGPU_HIDE_FROM_ABI
  constexpr iter_difference_t<_Ip> operator()(_Ip&& __first, _Sp __last) const {
    if constexpr (sized_sentinel_for<_Sp, std::remove_cv_t<std::remove_reference_t<_Ip>>>) {
      return __last - __first;
    } else {
      return __last - std::decay_t<_Ip>(__first);
    }
  }

  template<range _Rp>
  _LIBGPU_HIDE_FROM_ABI
  constexpr range_difference_t<_Rp> operator()(_Rp&& __r) const {
    if constexpr (sized_range<_Rp>) {
      return static_cast<range_difference_t<_Rp>>(ranges::size(__r));
    } else {
      return operator()(ranges::begin(__r), ranges::end(__r));
    }
  }
};

} // namespace __distance

inline namespace __cpo {
  inline constexpr auto distance = __distance::__fn{};
} // namespace __cpo
} // namespace ranges

#endif // _LIBGPU_STD_VER >= 20

} // namespace gpu

#endif // __GPU___ITERATOR_DISTANCE_H

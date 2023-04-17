//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_UNWRAP_RANGE_H
#define __GPU___ALGORITHM_UNWRAP_RANGE_H

#include "gpu/__config"

namespace gpu {

// __unwrap_range and __rewrap_range are used to unwrap ranges which may have different iterator and sentinel types.
// __unwrap_iter and __rewrap_iter don't work for this, because they assume that the iterator and sentinel have
// the same type. __unwrap_range tries to get two iterators and then forward to __unwrap_iter.

#if _LIBGPU_STD_VER >= 20
template <class _Iter, class _Sent>
struct __unwrap_range_impl {
  _LIBGPU_HIDE_FROM_ABI static constexpr auto __unwrap(_Iter __first, _Sent __sent)
    requires random_access_iterator<_Iter> && sized_sentinel_for<_Sent, _Iter>
  {
    auto __last = ranges::next(__first, __sent);
    return gpu::pair{gpu::__unwrap_iter(std::move(__first)), gpu::__unwrap_iter(std::move(__last))};
  }

  _LIBGPU_HIDE_FROM_ABI static constexpr auto __unwrap(_Iter __first, _Sent __last) {
    return gpu::pair{std::move(__first), std::move(__last)};
  }

  _LIBGPU_HIDE_FROM_ABI static constexpr auto
  __rewrap(_Iter __orig_iter, decltype(gpu::__unwrap_iter(std::move(__orig_iter))) __iter)
    requires random_access_iterator<_Iter> && sized_sentinel_for<_Sent, _Iter>
  {
    return gpu::__rewrap_iter(std::move(__orig_iter), std::move(__iter));
  }

  _LIBGPU_HIDE_FROM_ABI static constexpr auto __rewrap(const _Iter&, _Iter __iter)
    requires (!(random_access_iterator<_Iter> && sized_sentinel_for<_Sent, _Iter>))
  {
    return __iter;
  }
};

template <class _Iter>
struct __unwrap_range_impl<_Iter, _Iter> {
  _LIBGPU_HIDE_FROM_ABI static constexpr auto __unwrap(_Iter __first, _Iter __last) {
    return gpu::pair{gpu::__unwrap_iter(std::move(__first)), gpu::__unwrap_iter(std::move(__last))};
  }

  _LIBGPU_HIDE_FROM_ABI static constexpr auto
  __rewrap(_Iter __orig_iter, decltype(gpu::__unwrap_iter(__orig_iter)) __iter) {
    return gpu::__rewrap_iter(std::move(__orig_iter), std::move(__iter));
  }
};

template <class _Iter, class _Sent>
_LIBGPU_HIDE_FROM_ABI constexpr auto __unwrap_range(_Iter __first, _Sent __last) {
  return __unwrap_range_impl<_Iter, _Sent>::__unwrap(std::move(__first), std::move(__last));
}

template <
    class _Sent,
    class _Iter,
    class _Unwrapped = decltype(gpu::__unwrap_range(std::declval<_Iter>(), std::declval<_Sent>()))>
_LIBGPU_HIDE_FROM_ABI constexpr _Iter __rewrap_range(_Iter __orig_iter, _Unwrapped __iter) {
  return __unwrap_range_impl<_Iter, _Sent>::__rewrap(std::move(__orig_iter), std::move(__iter));
}
#else  // _LIBGPU_STD_VER >= 20
template <class _Iter, class _Unwrapped = decltype(gpu::__unwrap_iter(std::declval<_Iter>()))>
_LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR gpu::pair<_Unwrapped, _Unwrapped> __unwrap_range(_Iter __first, _Iter __last) {
  return gpu::make_pair(gpu::__unwrap_iter(std::move(__first)), gpu::__unwrap_iter(std::move(__last)));
}

template <class _Iter, class _Unwrapped = decltype(gpu::__unwrap_iter(std::declval<_Iter>()))>
_LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR _Iter __rewrap_range(_Iter __orig_iter, _Unwrapped __iter) {
  return gpu::__rewrap_iter(std::move(__orig_iter), std::move(__iter));
}
#endif // _LIBGPU_STD_VER >= 20

} // namespace gpu

#endif // __GPU___ALGORITHM_UNWRAP_RANGE_H

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_UNWRAP_ITER_H
#define __GPU___ALGORITHM_UNWRAP_ITER_H

#include "gpu/__config"
#include "gpu/__iterator/iterator_traits.h"

namespace gpu {

// TODO: Change the name of __unwrap_iter_impl to something more appropriate
// The job of __unwrap_iter is to remove iterator wrappers (like reverse_iterator or __wrap_iter),
// to reduce the number of template instantiations and to enable pointer-based optimizations e.g. in gpu::copy.
// In debug mode, we don't do this.
//
// Some algorithms (e.g. gpu::copy, but not gpu::sort) need to convert an
// "unwrapped" result back into the original iterator type. Doing that is the job of __rewrap_iter.

// Default case - we can't unwrap anything
template <class _Iter, bool = __is_cpp17_contiguous_iterator<_Iter>::value>
struct __unwrap_iter_impl {
  static _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR _Iter __rewrap(_Iter, _Iter __iter) { return __iter; }
  static _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR _Iter __unwrap(_Iter __i) _NOEXCEPT { return __i; }
};

#ifndef _LIBGPU_ENABLE_DEBUG_MODE

// It's a contiguous iterator, so we can use a raw pointer instead
template <class _Iter>
struct __unwrap_iter_impl<_Iter, true> {
  using _ToAddressT = decltype(std::__to_address(std::declval<_Iter>()));

  static _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR _Iter __rewrap(_Iter __orig_iter, _ToAddressT __unwrapped_iter) {
    return __orig_iter + (__unwrapped_iter - std::__to_address(__orig_iter));
  }

  static _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR _ToAddressT __unwrap(_Iter __i) _NOEXCEPT {
    return std::__to_address(__i);
  }
};

#endif // !_LIBGPU_ENABLE_DEBUG_MODE

template<class _Iter,
         class _Impl = __unwrap_iter_impl<_Iter>,
         std::enable_if_t<std::is_copy_constructible<_Iter>::value, int> = 0>
inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14
decltype(_Impl::__unwrap(std::declval<_Iter>())) __unwrap_iter(_Iter __i) _NOEXCEPT {
  return _Impl::__unwrap(__i);
}

// Allow input_iterators to be passed to __unwrap_iter (but not __rewrap_iter)
#if _LIBGPU_STD_VER >= 20
template <class _Iter, std::enable_if_t<!std::is_copy_constructible<_Iter>::value, int> = 0>
inline _LIBGPU_HIDE_FROM_ABI constexpr _Iter __unwrap_iter(_Iter __i) noexcept {
  return __i;
}
#endif

template <class _OrigIter, class _Iter, class _Impl = __unwrap_iter_impl<_OrigIter> >
_LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR _OrigIter __rewrap_iter(_OrigIter __orig_iter, _Iter __iter) _NOEXCEPT {
  return _Impl::__rewrap(std::move(__orig_iter), std::move(__iter));
}

} // namespace gpu

#endif // __GPU___ALGORITHM_UNWRAP_ITER_H

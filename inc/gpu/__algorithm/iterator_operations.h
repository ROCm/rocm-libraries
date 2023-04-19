//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_ITERATOR_OPERATIONS_H
#define __GPU___ALGORITHM_ITERATOR_OPERATIONS_H

#include "gpu/__config"
#include "gpu/__algorithm/iter_swap.h"

namespace gpu {

template <class _AlgPolicy> struct _IterOps;

#if _LIBGPU_STD_VER >= 20
struct _RangeAlgPolicy {};

template <>
struct _IterOps<_RangeAlgPolicy> {

  template <class _Iter>
  using __value_type = iter_value_t<_Iter>;

  template <class _Iter>
  using __iterator_category = ranges::__iterator_concept<_Iter>;

  template <class _Iter>
  using __difference_type = iter_difference_t<_Iter>;

  static constexpr auto advance = ranges::advance;
  static constexpr auto distance = ranges::distance;
  static constexpr auto __iter_move = ranges::iter_move;
  static constexpr auto iter_swap = ranges::iter_swap;
  static constexpr auto next = ranges::next;
  static constexpr auto prev = ranges::prev;
  static constexpr auto __advance_to = ranges::advance;
};

#endif

struct _ClassicAlgPolicy {};

template <>
struct _IterOps<_ClassicAlgPolicy> {

  template <class _Iter>
  using __value_type = typename std::iterator_traits<_Iter>::value_type;

  template <class _Iter>
  using __iterator_category = typename std::iterator_traits<_Iter>::iterator_category;

  template <class _Iter>
  using __difference_type = typename std::iterator_traits<_Iter>::difference_type;

  // advance
  template <class _Iter, class _Distance>
  __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14
  static void advance(_Iter& __iter, _Distance __count) {
    std::advance(__iter, __count);
  }

  // distance
  template <class _Iter>
  __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14
  static typename std::iterator_traits<_Iter>::difference_type distance(_Iter __first, _Iter __last) {
    return std::distance(__first, __last);
  }

  template <class _Iter>
  using __deref_t = decltype(*std::declval<_Iter&>());

  template <class _Iter>
  using __move_t = decltype(std::move(*std::declval<_Iter&>()));

  template <class _Iter>
  __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14
  static void __validate_iter_reference() {
    static_assert(std::is_same<__deref_t<_Iter>, typename std::iterator_traits<std::remove_cv_t<std::remove_reference_t<_Iter>> >::reference>::value,
        "It looks like your iterator's `std::iterator_traits<It>::reference` does not match the return type of "
        "dereferencing the iterator, i.e., calling `*it`. This is undefined behavior according to [input.iterators] "
        "and can lead to dangling reference issues at runtime, so we are flagging this.");
  }

  // iter_move
  template <class _Iter>
  __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14 static
  // If the result of dereferencing `_Iter` is a reference type, deduce the result of calling `std::move` on it. Note
  // that the C++03 mode doesn't support `decltype(auto)` as the return type.
  std::enable_if_t<
      std::is_reference<__deref_t<_Iter> >::value,
      __move_t<_Iter> >
  __iter_move(_Iter&& __i) {
    __validate_iter_reference<_Iter>();

    return std::move(*std::forward<_Iter>(__i));
  }

  template <class _Iter>
  __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14 static
  // If the result of dereferencing `_Iter` is a value type, deduce the return value of this function to also be a
  // value -- otherwise, after `operator*` returns a temporary, this function would return a dangling reference to that
  // temporary. Note that the C++03 mode doesn't support `auto` as the return type.
  std::enable_if_t<
      !std::is_reference<__deref_t<_Iter> >::value,
      __deref_t<_Iter> >
  __iter_move(_Iter&& __i) {
    __validate_iter_reference<_Iter>();

    return *std::forward<_Iter>(__i);
  }

  // iter_swap
  template <class _Iter1, class _Iter2>
  __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14
  static void iter_swap(_Iter1&& __a, _Iter2&& __b) {
    gpu::iter_swap(std::forward<_Iter1>(__a), std::forward<_Iter2>(__b));
  }

  // next
  template <class _Iterator>
  __device__ _LIBGPU_HIDE_FROM_ABI static _LIBGPU_CONSTEXPR_SINCE_CXX14
  _Iterator next(_Iterator, _Iterator __last) {
    return __last;
  }

  template <class _Iter>
  __device__ _LIBGPU_HIDE_FROM_ABI static _LIBGPU_CONSTEXPR_SINCE_CXX14
  std::remove_cv_t<std::remove_reference_t<_Iter>> next(_Iter&& __it,
                          typename std::iterator_traits<std::remove_cv_t<std::remove_reference_t<_Iter>> >::difference_type __n = 1) {
    return std::next(std::forward<_Iter>(__it), __n);
  }

  // prev
  template <class _Iter>
  __device__ _LIBGPU_HIDE_FROM_ABI static _LIBGPU_CONSTEXPR_SINCE_CXX14
  std::remove_cv_t<std::remove_reference_t<_Iter>> prev(_Iter&& __iter,
                 typename std::iterator_traits<std::remove_cv_t<std::remove_reference_t<_Iter>> >::difference_type __n = 1) {
    return std::prev(std::forward<_Iter>(__iter), __n);
  }

  template <class _Iter>
  __device__ _LIBGPU_HIDE_FROM_ABI static _LIBGPU_CONSTEXPR_SINCE_CXX14
  void __advance_to(_Iter& __first, _Iter __last) {
    __first = __last;
  }
};

} // namespace gpu

#endif // __GPU___ALGORITHM_ITERATOR_OPERATIONS_H

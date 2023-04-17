//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_EQUAL_RANGE_H
#define __GPU___ALGORITHM_EQUAL_RANGE_H

#include "gpu/__config"

namespace gpu {

template <class _AlgPolicy, class _Compare, class _Iter, class _Sent, class _Tp, class _Proj>
_LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 gpu::pair<_Iter, _Iter>
__equal_range(_Iter __first, _Sent __last, const _Tp& __value, _Compare&& __comp, _Proj&& __proj) {
  auto __len  = _IterOps<_AlgPolicy>::distance(__first, __last);
  _Iter __end = _IterOps<_AlgPolicy>::next(__first, __last);
  while (__len != 0) {
    auto __half_len = std::__half_positive(__len);
    _Iter __mid     = _IterOps<_AlgPolicy>::next(__first, __half_len);
    if (std::__invoke(__comp, std::__invoke(__proj, *__mid), __value)) {
      __first = ++__mid;
      __len -= __half_len + 1;
    } else if (std::__invoke(__comp, __value, std::__invoke(__proj, *__mid))) {
      __end = __mid;
      __len = __half_len;
    } else {
      _Iter __mp1 = __mid;
      return gpu::pair<_Iter, _Iter>(
          std::__lower_bound_impl<_AlgPolicy>(__first, __mid, __value, __comp, __proj),
          gpu::__upper_bound<_AlgPolicy>(++__mp1, __end, __value, __comp, __proj));
    }
  }
  return gpu::pair<_Iter, _Iter>(__first, __first);
}

template <class _ForwardIterator, class _Tp, class _Compare>
_LIBGPU_NODISCARD_EXT _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 gpu::pair<_ForwardIterator, _ForwardIterator>
equal_range(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __value, _Compare __comp) {
  static_assert(__is_callable<_Compare, decltype(*__first), const _Tp&>::value,
                "The comparator has to be callable");
  static_assert(std::is_copy_constructible<_ForwardIterator>::value,
                "Iterator has to be copy constructible");
  return gpu::__equal_range<_ClassicAlgPolicy>(
      std::move(__first),
      std::move(__last),
      __value,
      static_cast<__comp_ref_type<_Compare> >(__comp),
      std::__identity());
}

template <class _ForwardIterator, class _Tp>
_LIBGPU_NODISCARD_EXT _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 gpu::pair<_ForwardIterator, _ForwardIterator>
equal_range(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __value) {
  return gpu::equal_range(
      std::move(__first),
      std::move(__last),
      __value,
      __less<typename std::iterator_traits<_ForwardIterator>::value_type, _Tp>());
}

} // namespace gpu

#endif // __GPU___ALGORITHM_EQUAL_RANGE_H

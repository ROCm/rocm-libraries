//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_PARTIAL_SORT_H
#define __GPU___ALGORITHM_PARTIAL_SORT_H

#include "gpu/__config"

namespace gpu {

template <class _AlgPolicy, class _Compare, class _RandomAccessIterator, class _Sentinel>
__device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20
_RandomAccessIterator __partial_sort_impl(
    _RandomAccessIterator __first, _RandomAccessIterator __middle, _Sentinel __last, _Compare&& __comp) {
  if (__first == __middle) {
    return _IterOps<_AlgPolicy>::next(__middle, __last);
  }

  gpu::__make_heap<_AlgPolicy>(__first, __middle, __comp);

  typename std::iterator_traits<_RandomAccessIterator>::difference_type __len = __middle - __first;
  _RandomAccessIterator __i = __middle;
  for (; __i != __last; ++__i)
  {
      if (__comp(*__i, *__first))
      {
          _IterOps<_AlgPolicy>::iter_swap(__i, __first);
          std::__sift_down<_AlgPolicy>(__first, __comp, __len, __first);
      }
  }
  gpu::__sort_heap<_AlgPolicy>(std::move(__first), std::move(__middle), __comp);

  return __i;
}

template <class _AlgPolicy, class _Compare, class _RandomAccessIterator, class _Sentinel>
__device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20
_RandomAccessIterator __partial_sort(_RandomAccessIterator __first, _RandomAccessIterator __middle, _Sentinel __last,
                                     _Compare& __comp) {
  if (__first == __middle)
      return _IterOps<_AlgPolicy>::next(__middle, __last);

  std::__debug_randomize_range<_AlgPolicy>(__first, __last);

  auto __last_iter =
      std::__partial_sort_impl<_AlgPolicy>(__first, __middle, __last, static_cast<__comp_ref_type<_Compare> >(__comp));

  std::__debug_randomize_range<_AlgPolicy>(__middle, __last);

  return __last_iter;
}

template <class _RandomAccessIterator, class _Compare>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
void
partial_sort(_RandomAccessIterator __first, _RandomAccessIterator __middle, _RandomAccessIterator __last,
             _Compare __comp)
{
  static_assert(std::is_copy_constructible<_RandomAccessIterator>::value, "Iterators must be copy constructible.");
  static_assert(std::is_copy_assignable<_RandomAccessIterator>::value, "Iterators must be copy assignable.");

  (void)gpu::__partial_sort<_ClassicAlgPolicy>(std::move(__first), std::move(__middle), std::move(__last), __comp);
}

template <class _RandomAccessIterator>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
void
partial_sort(_RandomAccessIterator __first, _RandomAccessIterator __middle, _RandomAccessIterator __last)
{
    gpu::partial_sort(__first, __middle, __last,
                        __less<typename std::iterator_traits<_RandomAccessIterator>::value_type>());
}

} // namespace gpu

#endif // __GPU___ALGORITHM_PARTIAL_SORT_H

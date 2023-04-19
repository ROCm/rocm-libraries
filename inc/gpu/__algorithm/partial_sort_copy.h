//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_PARTIAL_SORT_COPY_H
#define __GPU___ALGORITHM_PARTIAL_SORT_COPY_H

#include "gpu/__config"

namespace gpu {

template <class _AlgPolicy, class _Compare,
          class _InputIterator, class _Sentinel1, class _RandomAccessIterator, class _Sentinel2,
          class _Proj1, class _Proj2>
__device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 gpu::pair<_InputIterator, _RandomAccessIterator>
__partial_sort_copy(_InputIterator __first, _Sentinel1 __last,
                    _RandomAccessIterator __result_first, _Sentinel2 __result_last,
                    _Compare&& __comp, _Proj1&& __proj1, _Proj2&& __proj2)
{
    _RandomAccessIterator __r = __result_first;
    auto&& __projected_comp = std::__make_projected(__comp, __proj2);

    if (__r != __result_last)
    {
        for (; __first != __last && __r != __result_last; ++__first, (void) ++__r)
            *__r = *__first;
        gpu::__make_heap<_AlgPolicy>(__result_first, __r, __projected_comp);
        typename std::iterator_traits<_RandomAccessIterator>::difference_type __len = __r - __result_first;
        for (; __first != __last; ++__first)
            if (std::__invoke(__comp, std::__invoke(__proj1, *__first), std::__invoke(__proj2, *__result_first))) {
                *__result_first = *__first;
                std::__sift_down<_AlgPolicy>(__result_first, __projected_comp, __len, __result_first);
            }
        gpu::__sort_heap<_AlgPolicy>(__result_first, __r, __projected_comp);
    }

    return gpu::pair<_InputIterator, _RandomAccessIterator>(
        _IterOps<_AlgPolicy>::next(std::move(__first), std::move(__last)), std::move(__r));
}

template <class _InputIterator, class _RandomAccessIterator, class _Compare>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
_RandomAccessIterator
partial_sort_copy(_InputIterator __first, _InputIterator __last,
                  _RandomAccessIterator __result_first, _RandomAccessIterator __result_last, _Compare __comp)
{
  static_assert(__is_callable<_Compare, decltype(*__first), decltype(*__result_first)>::value,
                "Comparator has to be callable");

  auto __result = gpu::__partial_sort_copy<_ClassicAlgPolicy>(__first, __last, __result_first, __result_last,
      static_cast<__comp_ref_type<_Compare> >(__comp), __identity(), __identity());
  return __result.second;
}

template <class _InputIterator, class _RandomAccessIterator>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
_RandomAccessIterator
partial_sort_copy(_InputIterator __first, _InputIterator __last,
                  _RandomAccessIterator __result_first, _RandomAccessIterator __result_last)
{
    return gpu::partial_sort_copy(__first, __last, __result_first, __result_last,
                                   __less<typename std::iterator_traits<_RandomAccessIterator>::value_type>());
}

} // namespace gpu

#endif // __GPU___ALGORITHM_PARTIAL_SORT_COPY_H

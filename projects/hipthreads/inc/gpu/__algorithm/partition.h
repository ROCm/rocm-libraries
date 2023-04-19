//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_PARTITION_H
#define __GPU___ALGORITHM_PARTITION_H

#include "gpu/__config"

namespace gpu {

template <class _Predicate, class _AlgPolicy, class _ForwardIterator, class _Sentinel>
__device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 gpu::pair<_ForwardIterator, _ForwardIterator>
__partition_impl(_ForwardIterator __first, _Sentinel __last, _Predicate __pred, std::forward_iterator_tag)
{
    while (true)
    {
        if (__first == __last)
            return gpu::make_pair(std::move(__first), std::move(__first));
        if (!__pred(*__first))
            break;
        ++__first;
    }

    _ForwardIterator __p = __first;
    while (++__p != __last)
    {
        if (__pred(*__p))
        {
            _IterOps<_AlgPolicy>::iter_swap(__first, __p);
            ++__first;
        }
    }
    return gpu::make_pair(std::move(__first), std::move(__p));
}

template <class _Predicate, class _AlgPolicy, class _BidirectionalIterator, class _Sentinel>
__device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 gpu::pair<_BidirectionalIterator, _BidirectionalIterator>
__partition_impl(_BidirectionalIterator __first, _Sentinel __sentinel, _Predicate __pred,
            std::bidirectional_iterator_tag)
{
    _BidirectionalIterator __original_last = _IterOps<_AlgPolicy>::next(__first, __sentinel);
    _BidirectionalIterator __last = __original_last;

    while (true)
    {
        while (true)
        {
            if (__first == __last)
                return gpu::make_pair(std::move(__first), std::move(__original_last));
            if (!__pred(*__first))
                break;
            ++__first;
        }
        do
        {
            if (__first == --__last)
                return gpu::make_pair(std::move(__first), std::move(__original_last));
        } while (!__pred(*__last));
        _IterOps<_AlgPolicy>::iter_swap(__first, __last);
        ++__first;
    }
}

template <class _AlgPolicy, class _ForwardIterator, class _Sentinel, class _Predicate, class _IterCategory>
__device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20
gpu::pair<_ForwardIterator, _ForwardIterator> __partition(
    _ForwardIterator __first, _Sentinel __last, _Predicate&& __pred, _IterCategory __iter_category) {
  return std::__partition_impl<std::remove_cv_t<std::remove_reference_t<_Predicate>>&, _AlgPolicy>(
      std::move(__first), std::move(__last), __pred, __iter_category);
}

template <class _ForwardIterator, class _Predicate>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
_ForwardIterator
partition(_ForwardIterator __first, _ForwardIterator __last, _Predicate __pred)
{
  using _IterCategory = typename std::iterator_traits<_ForwardIterator>::iterator_category;
  auto __result = gpu::__partition<_ClassicAlgPolicy>(std::move(__first), std::move(__last), __pred, _IterCategory());
  return __result.first;
}

} // namespace gpu

#endif // __GPU___ALGORITHM_PARTITION_H

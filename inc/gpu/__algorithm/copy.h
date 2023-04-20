//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_COPY_H
#define __GPU___ALGORITHM_COPY_H

#include "gpu/__config"
#include "gpu/__algorithm/copy_move_common.h"
#include "gpu/__algorithm/min.h"
#include "gpu/__iterator/iterator_traits.h"
#include "gpu/__iterator/segmented_iterator.h"
#include "gpu/__utility/pair.h"
#include "gpu/__utility/swap.h"

namespace gpu {

template <class, class _InIter, class _Sent, class _OutIter>
__device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14 gpu::pair<_InIter, _OutIter> __copy(_InIter, _Sent, _OutIter);

template <class _AlgPolicy>
struct __copy_loop {
  template <class _InIter, class _Sent, class _OutIter>
  __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14 gpu::pair<_InIter, _OutIter>
  operator()(_InIter __first, _Sent __last, _OutIter __result) const {
    while (__first != __last) {
      *__result = *__first;
      ++__first;
      ++__result;
    }

    return gpu::make_pair(std::move(__first), std::move(__result));
  }

#if 0 // TODO: figure out why SFINAE isn't stopping this from being used
  template <class _InIter, class _OutIter, std::enable_if_t<__is_segmented_iterator<_InIter>::value, int> = 0>
  __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14 gpu::pair<_InIter, _OutIter>
  operator()(_InIter __first, _InIter __last, _OutIter __result) const {
    using _Traits = __segmented_iterator_traits<_InIter>;
    auto __sfirst = _Traits::__segment(__first);
    auto __slast  = _Traits::__segment(__last);
    if (__sfirst == __slast) {
      auto __iters = gpu::__copy<_AlgPolicy>(_Traits::__local(__first), _Traits::__local(__last), std::move(__result));
      return gpu::make_pair(__last, std::move(__iters.second));
    }

    __result = gpu::__copy<_AlgPolicy>(_Traits::__local(__first), _Traits::__end(__sfirst), std::move(__result)).second;
    ++__sfirst;
    while (__sfirst != __slast) {
      __result =
          gpu::__copy<_AlgPolicy>(_Traits::__begin(__sfirst), _Traits::__end(__sfirst), std::move(__result)).second;
      ++__sfirst;
    }
    __result =
        gpu::__copy<_AlgPolicy>(_Traits::__begin(__sfirst), _Traits::__local(__last), std::move(__result)).second;
    return gpu::make_pair(__last, std::move(__result));
  }

  template <class _InIter,
            class _OutIter,
            std::enable_if_t<__is_cpp17_random_access_iterator<_InIter>::value &&
                              !__is_segmented_iterator<_InIter>::value && __is_segmented_iterator<_OutIter>::value,
                          int> = 0>
  __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14 gpu::pair<_InIter, _OutIter>
  operator()(_InIter __first, _InIter __last, _OutIter __result) {
    using _Traits = __segmented_iterator_traits<_OutIter>;
    using _DiffT  = typename std::common_type<gpu::__iter_diff_t<_InIter>, gpu::__iter_diff_t<_OutIter> >::type;

    if (__first == __last)
      return gpu::make_pair(std::move(__first), std::move(__result));

    auto __local_first      = _Traits::__local(__result);
    auto __segment_iterator = _Traits::__segment(__result);
    while (true) {
      auto __local_last = _Traits::__end(__segment_iterator);
      auto __size       = gpu::min<_DiffT>(__local_last - __local_first, __last - __first);
      auto __iters      = gpu::__copy<_AlgPolicy>(__first, __first + __size, __local_first);
      __first           = std::move(__iters.first);

      if (__first == __last)
        return gpu::make_pair(std::move(__first), _Traits::__compose(__segment_iterator, std::move(__iters.second)));

      __local_first = _Traits::__begin(++__segment_iterator);
    }
  }
#endif // 0
};

struct __copy_trivial {
  // At this point, the iterators have been unwrapped so any `contiguous_iterator` has been unwrapped to a pointer.
  template <class _In, class _Out,
            std::enable_if_t<__can_lower_copy_assignment_to_memmove<_In, _Out>::value, int> = 0>
  __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14 gpu::pair<_In*, _Out*>
  operator()(_In* __first, _In* __last, _Out* __result) const {
    return gpu::__copy_trivial_impl(__first, __last, __result);
  }
};

template <class _AlgPolicy, class _InIter, class _Sent, class _OutIter>
__device__ gpu::pair<_InIter, _OutIter> inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14
__copy(_InIter __first, _Sent __last, _OutIter __result) {
  return gpu::__dispatch_copy_or_move<_AlgPolicy, __copy_loop<_AlgPolicy>, __copy_trivial>(
      std::move(__first), std::move(__last), std::move(__result));
}

template <class _InputIterator, class _OutputIterator>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20 _OutputIterator
copy(_InputIterator __first, _InputIterator __last, _OutputIterator __result) {
  return gpu::__copy<_ClassicAlgPolicy>(__first, __last, __result).second;
}

} // namespace gpu

#endif // __GPU___ALGORITHM_COPY_H

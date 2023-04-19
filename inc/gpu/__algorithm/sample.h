//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_SAMPLE_H
#define __GPU___ALGORITHM_SAMPLE_H

#include "gpu/__config"

namespace gpu {

template <class _AlgPolicy,
          class _PopulationIterator, class _PopulationSentinel, class _SampleIterator, class _Distance,
          class _UniformRandomNumberGenerator>
__device__ _LIBGPU_INLINE_VISIBILITY
_SampleIterator __sample(_PopulationIterator __first,
                         _PopulationSentinel __last, _SampleIterator __output_iter,
                         _Distance __n,
                         _UniformRandomNumberGenerator& __g,
                         std::input_iterator_tag) {

  _Distance __k = 0;
  for (; __first != __last && __k < __n; ++__first, (void) ++__k)
    __output_iter[__k] = *__first;
  _Distance __sz = __k;
  for (; __first != __last; ++__first, (void) ++__k) {
    _Distance __r = uniform_int_distribution<_Distance>(0, __k)(__g);
    if (__r < __sz)
      __output_iter[__r] = *__first;
  }
  return __output_iter + gpu::min(__n, __k);
}

template <class _AlgPolicy,
          class _PopulationIterator, class _PopulationSentinel, class _SampleIterator, class _Distance,
          class _UniformRandomNumberGenerator>
__device__ _LIBGPU_INLINE_VISIBILITY
_SampleIterator __sample(_PopulationIterator __first,
                         _PopulationSentinel __last, _SampleIterator __output_iter,
                         _Distance __n,
                         _UniformRandomNumberGenerator& __g,
                         std::forward_iterator_tag) {
  _Distance __unsampled_sz = _IterOps<_AlgPolicy>::distance(__first, __last);
  for (__n = gpu::min(__n, __unsampled_sz); __n != 0; ++__first) {
    _Distance __r = uniform_int_distribution<_Distance>(0, --__unsampled_sz)(__g);
    if (__r < __n) {
      *__output_iter++ = *__first;
      --__n;
    }
  }
  return __output_iter;
}

template <class _AlgPolicy,
          class _PopulationIterator, class _PopulationSentinel, class _SampleIterator, class _Distance,
          class _UniformRandomNumberGenerator>
__device__ _LIBGPU_INLINE_VISIBILITY
_SampleIterator __sample(_PopulationIterator __first,
                         _PopulationSentinel __last, _SampleIterator __output_iter,
                         _Distance __n, _UniformRandomNumberGenerator& __g) {
  _LIBGPU_ASSERT(__n >= 0, "N must be a positive number.");

  using _PopIterCategory = typename _IterOps<_AlgPolicy>::template __iterator_category<_PopulationIterator>;
  using _Difference = typename _IterOps<_AlgPolicy>::template __difference_type<_PopulationIterator>;
  using _CommonType = typename std::common_type<_Distance, _Difference>::type;

  return gpu::__sample<_AlgPolicy>(
      std::move(__first), std::move(__last), std::move(__output_iter), _CommonType(__n),
      __g, _PopIterCategory());
}

#if _LIBGPU_STD_VER >= 17
template <class _PopulationIterator, class _SampleIterator, class _Distance,
          class _UniformRandomNumberGenerator>
__device__ inline _LIBGPU_INLINE_VISIBILITY
_SampleIterator sample(_PopulationIterator __first,
                       _PopulationIterator __last, _SampleIterator __output_iter,
                       _Distance __n, _UniformRandomNumberGenerator&& __g) {
  static_assert(__is_cpp17_forward_iterator<_PopulationIterator>::value ||
                __is_cpp17_random_access_iterator<_SampleIterator>::value,
                "SampleIterator must meet the requirements of RandomAccessIterator");

  return gpu::__sample<_ClassicAlgPolicy>(
      std::move(__first), std::move(__last), std::move(__output_iter), __n, __g);
}

#endif // _LIBGPU_STD_VER >= 17

} // namespace gpu

#endif // __GPU___ALGORITHM_SAMPLE_H

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_MIN_ELEMENT_H
#define __GPU___ALGORITHM_MIN_ELEMENT_H

#include "gpu/__config"
#include "gpu/__functional/identity.h"
#include "gpu/__iterator/iterator_traits.h"

namespace gpu {

template <class _Comp, class _Iter, class _Sent, class _Proj>
inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14
_Iter __min_element(_Iter __first, _Sent __last, _Comp __comp, _Proj& __proj) {
  if (__first == __last)
    return __first;

  _Iter __i = __first;
  while (++__i != __last)
    if (std::__invoke(__comp, std::__invoke(__proj, *__i), std::__invoke(__proj, *__first)))
      __first = __i;

  return __first;
}

template <class _Comp, class _Iter, class _Sent>
_LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14
_Iter __min_element(_Iter __first, _Sent __last, _Comp __comp) {
  auto __proj = __identity();
  return gpu::__min_element<_Comp>(std::move(__first), std::move(__last), __comp, __proj);
}

template <class _ForwardIterator, class _Compare>
_LIBGPU_NODISCARD_EXT inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14 _ForwardIterator
min_element(_ForwardIterator __first, _ForwardIterator __last, _Compare __comp)
{
  static_assert(__is_cpp17_forward_iterator<_ForwardIterator>::value,
      "gpu::min_element requires a ForwardIterator");

  return gpu::__min_element<__comp_ref_type<_Compare> >(std::move(__first), std::move(__last), __comp);
}

template <class _ForwardIterator>
_LIBGPU_NODISCARD_EXT inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14 _ForwardIterator
min_element(_ForwardIterator __first, _ForwardIterator __last)
{
    return gpu::min_element(__first, __last,
              __less<typename std::iterator_traits<_ForwardIterator>::value_type>());
}

} // namespace gpu

#endif // __GPU___ALGORITHM_MIN_ELEMENT_H

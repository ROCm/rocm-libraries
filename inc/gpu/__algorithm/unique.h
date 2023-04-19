//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_UNIQUE_H
#define __GPU___ALGORITHM_UNIQUE_H

#include "gpu/__config"

namespace gpu {

// unique

template <class _AlgPolicy, class _Iter, class _Sent, class _BinaryPredicate>
_LIBGPU_NODISCARD_EXT __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 gpu::pair<_Iter, _Iter>
__unique(_Iter __first, _Sent __last, _BinaryPredicate&& __pred) {
  __first = gpu::__adjacent_find(__first, __last, __pred);
  if (__first != __last) {
    // ...  a  a  ?  ...
    //      f     i
    _Iter __i = __first;
    for (++__i; ++__i != __last;)
      if (!__pred(*__first, *__i))
        *++__first = _IterOps<_AlgPolicy>::__iter_move(__i);
    ++__first;
    return gpu::pair<_Iter, _Iter>(std::move(__first), std::move(__i));
  }
  return gpu::pair<_Iter, _Iter>(__first, __first);
}

template <class _ForwardIterator, class _BinaryPredicate>
_LIBGPU_NODISCARD_EXT __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 _ForwardIterator
unique(_ForwardIterator __first, _ForwardIterator __last, _BinaryPredicate __pred) {
  return gpu::__unique<_ClassicAlgPolicy>(std::move(__first), std::move(__last), __pred).first;
}

template <class _ForwardIterator>
_LIBGPU_NODISCARD_EXT __device__ inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 _ForwardIterator
unique(_ForwardIterator __first, _ForwardIterator __last) {
  return gpu::unique(__first, __last, __equal_to());
}

} // namespace gpu

#endif // __GPU___ALGORITHM_UNIQUE_H

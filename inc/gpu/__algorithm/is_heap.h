//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_IS_HEAP_H
#define __GPU___ALGORITHM_IS_HEAP_H

#include "gpu/__config"

namespace gpu {

template <class _RandomAccessIterator, class _Compare>
_LIBGPU_NODISCARD_EXT __device__ inline
_LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
bool
is_heap(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp)
{
    return gpu::__is_heap_until(__first, __last, static_cast<__comp_ref_type<_Compare> >(__comp)) == __last;
}

template<class _RandomAccessIterator>
_LIBGPU_NODISCARD_EXT __device__ inline
_LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
bool
is_heap(_RandomAccessIterator __first, _RandomAccessIterator __last)
{
    return gpu::is_heap(__first, __last, __less<typename std::iterator_traits<_RandomAccessIterator>::value_type>());
}

} // namespace gpu

#endif // __GPU___ALGORITHM_IS_HEAP_H

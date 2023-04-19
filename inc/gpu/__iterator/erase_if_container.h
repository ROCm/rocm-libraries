// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ITERATOR_ERASE_IF_CONTAINER_H
#define __GPU___ITERATOR_ERASE_IF_CONTAINER_H

#include "gpu/__config"

namespace gpu {

template <class _Container, class _Predicate>
__device__ _LIBGPU_HIDE_FROM_ABI
typename _Container::size_type
__libgpu_erase_if_container(_Container& __c, _Predicate& __pred) {
  typename _Container::size_type __old_size = __c.size();

  const typename _Container::iterator __last = __c.end();
  for (typename _Container::iterator __iter = __c.begin(); __iter != __last;) {
    if (__pred(*__iter))
      __iter = __c.erase(__iter);
    else
      ++__iter;
  }

  return __old_size - __c.size();
}

} // namespace gpu

#endif // __GPU___ITERATOR_ERASE_IF_CONTAINER_H

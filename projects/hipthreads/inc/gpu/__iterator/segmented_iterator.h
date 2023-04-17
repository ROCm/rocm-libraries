//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___SEGMENTED_ITERATOR_H
#define __GPU___SEGMENTED_ITERATOR_H

#include "gpu/__config"

namespace gpu {

template <class _Iterator>
struct __segmented_iterator_traits;
/* exposition-only:
{
  using __segment_iterator = ...;
  using __local_iterator   = ...;

  static __segment_iterator __segment(_Iterator);
  static __local_iterator __local(_Iterator);
  static __local_iterator __begin(__segment_iterator);
  static __local_iterator __end(__segment_iterator);
  static _Iterator __compose(__segment_iterator, __local_iterator);
};
*/

template <class _Tp, std::size_t = 0>
struct __has_specialization : std::false_type {};

template <class _Tp>
struct __has_specialization<_Tp, sizeof(_Tp) * 0> : std::true_type {};

template <class _Iterator>
using __is_segmented_iterator = __has_specialization<__segmented_iterator_traits<_Iterator> >;

} // namespace gpu

#endif // __GPU___SEGMENTED_ITERATOR_H

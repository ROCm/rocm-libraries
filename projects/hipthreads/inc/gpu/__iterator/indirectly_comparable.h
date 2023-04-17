// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ITERATOR_INDIRECTLY_COMPARABLE_H
#define __GPU___ITERATOR_INDIRECTLY_COMPARABLE_H

#include "gpu/__config"

namespace gpu {

#if _LIBGPU_STD_VER >= 20

template <class _I1, class _I2, class _Rp, class _P1 = identity, class _P2 = identity>
concept indirectly_comparable =
  indirect_binary_predicate<_Rp, projected<_I1, _P1>, projected<_I2, _P2>>;

#endif // _LIBGPU_STD_VER >= 20

} // namespace gpu

#endif // __GPU___ITERATOR_INDIRECTLY_COMPARABLE_H

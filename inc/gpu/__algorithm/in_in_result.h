// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_IN_IN_RESULT_H
#define __GPU___ALGORITHM_IN_IN_RESULT_H

#include "gpu/__config"

namespace gpu {

#if _LIBGPU_STD_VER >= 20

namespace ranges {

template <class _InIter1, class _InIter2>
struct in_in_result {
  _LIBGPU_NO_UNIQUE_ADDRESS _InIter1 in1;
  _LIBGPU_NO_UNIQUE_ADDRESS _InIter2 in2;

  template <class _InIter3, class _InIter4>
    requires std::convertible_to<const _InIter1&, _InIter3> && std::convertible_to<const _InIter2&, _InIter4>
   _LIBGPU_HIDE_FROM_ABI constexpr
   operator in_in_result<_InIter3, _InIter4>() const & {
    return {in1, in2};
  }

  template <class _InIter3, class _InIter4>
    requires std::convertible_to<_InIter1, _InIter3> && std::convertible_to<_InIter2, _InIter4>
  _LIBGPU_HIDE_FROM_ABI constexpr
  operator in_in_result<_InIter3, _InIter4>() && {
    return {std::move(in1), std::move(in2)};
  }
};

} // namespace ranges

#endif // _LIBGPU_STD_VER >= 20

} // namespace gpu

#endif // __GPU___ALGORITHM_IN_IN_RESULT_H

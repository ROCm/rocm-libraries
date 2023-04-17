// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_IN_IN_OUT_RESULT_H
#define __GPU___ALGORITHM_IN_IN_OUT_RESULT_H

#include "gpu/__config"

namespace gpu {

#if _LIBGPU_STD_VER >= 20

namespace ranges {

template <class _InIter1, class _InIter2, class _OutIter1>
struct in_in_out_result {
  _LIBGPU_NO_UNIQUE_ADDRESS _InIter1 in1;
  _LIBGPU_NO_UNIQUE_ADDRESS _InIter2 in2;
  _LIBGPU_NO_UNIQUE_ADDRESS _OutIter1 out;

  template <class _InIter3, class _InIter4, class _OutIter2>
    requires std::convertible_to<const _InIter1&, _InIter3>
          && std::convertible_to<const _InIter2&, _InIter4> && std::convertible_to<const _OutIter1&, _OutIter2>
  _LIBGPU_HIDE_FROM_ABI constexpr
  operator in_in_out_result<_InIter3, _InIter4, _OutIter2>() const& {
    return {in1, in2, out};
  }

  template <class _InIter3, class _InIter4, class _OutIter2>
    requires std::convertible_to<_InIter1, _InIter3>
          && std::convertible_to<_InIter2, _InIter4> && std::convertible_to<_OutIter1, _OutIter2>
  _LIBGPU_HIDE_FROM_ABI constexpr
  operator in_in_out_result<_InIter3, _InIter4, _OutIter2>() && {
    return {std::move(in1), std::move(in2), std::move(out)};
  }
};

} // namespace ranges

#endif // _LIBGPU_STD_VER >= 20

} // namespace gpu

#endif // __GPU___ALGORITHM_IN_IN_OUT_RESULT_H

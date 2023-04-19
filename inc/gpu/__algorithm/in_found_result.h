// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_IN_FOUND_RESULT_H
#define __GPU___ALGORITHM_IN_FOUND_RESULT_H

#include "gpu/__config"

namespace gpu {

namespace ranges {
template <class _InIter1>
struct in_found_result {
  _LIBGPU_NO_UNIQUE_ADDRESS _InIter1 in;
  bool found;

  template <class _InIter2>
    requires std::convertible_to<const _InIter1&, _InIter2>
  __device__ _LIBGPU_HIDE_FROM_ABI constexpr operator in_found_result<_InIter2>() const & {
    return {in, found};
  }

  template <class _InIter2>
    requires std::convertible_to<_InIter1, _InIter2>
  __device__ _LIBGPU_HIDE_FROM_ABI constexpr operator in_found_result<_InIter2>() && {
    return {std::move(in), found};
  }
};
} // namespace ranges

} // namespace gpu

#endif // __GPU___ALGORITHM_IN_FOUND_RESULT_H

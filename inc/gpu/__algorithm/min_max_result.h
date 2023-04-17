// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_MIN_MAX_RESULT_H
#define __GPU___ALGORITHM_MIN_MAX_RESULT_H

#include "gpu/__config"

namespace gpu {

#if _LIBGPU_STD_VER >= 20

namespace ranges {

template <class _T1>
struct min_max_result {
  _LIBGPU_NO_UNIQUE_ADDRESS _T1 min;
  _LIBGPU_NO_UNIQUE_ADDRESS _T1 max;

  template <class _T2>
    requires std::convertible_to<const _T1&, _T2>
  _LIBGPU_HIDE_FROM_ABI constexpr operator min_max_result<_T2>() const & {
    return {min, max};
  }

  template <class _T2>
    requires std::convertible_to<_T1, _T2>
  _LIBGPU_HIDE_FROM_ABI constexpr operator min_max_result<_T2>() && {
    return {std::move(min), std::move(max)};
  }
};

} // namespace ranges

#endif // _LIBGPU_STD_VER >= 20

} // namespace gpu

#endif // __GPU___ALGORITHM_MIN_MAX_RESULT_H

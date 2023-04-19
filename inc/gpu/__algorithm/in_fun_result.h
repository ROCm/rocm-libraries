// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_IN_FUN_RESULT_H
#define __GPU___ALGORITHM_IN_FUN_RESULT_H

#include "gpu/__config"

namespace gpu {

#if _LIBGPU_STD_VER >= 20

namespace ranges {
template <class _InIter1, class _Func1>
struct in_fun_result {
  _LIBGPU_NO_UNIQUE_ADDRESS _InIter1 in;
  _LIBGPU_NO_UNIQUE_ADDRESS _Func1 fun;

  template <class _InIter2, class _Func2>
    requires std::convertible_to<const _InIter1&, _InIter2> && std::convertible_to<const _Func1&, _Func2>
  __device__ _LIBGPU_HIDE_FROM_ABI constexpr operator in_fun_result<_InIter2, _Func2>() const & {
    return {in, fun};
  }

  template <class _InIter2, class _Func2>
    requires std::convertible_to<_InIter1, _InIter2> && std::convertible_to<_Func1, _Func2>
  __device__ _LIBGPU_HIDE_FROM_ABI constexpr operator in_fun_result<_InIter2, _Func2>() && {
    return {std::move(in), std::move(fun)};
  }
};
} // namespace ranges

#endif // _LIBGPU_STD_VER >= 20

} // namespace gpu

#endif // __GPU___ALGORITHM_IN_FUN_RESULT_H

// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef __GPU___ALGORITHM_MAKE_PROJECTED_H
#define __GPU___ALGORITHM_MAKE_PROJECTED_H

#include "gpu/__config"

namespace gpu {

template <class _Pred, class _Proj>
struct _ProjectedPred {
  _Pred& __pred; // Can be a unary or a binary predicate.
  _Proj& __proj;

  _LIBGPU_CONSTEXPR _ProjectedPred(_Pred& __pred_arg, _Proj& __proj_arg) : __pred(__pred_arg), __proj(__proj_arg) {}

  template <class _Tp>
  typename __invoke_of<_Pred&,
                       decltype(std::__invoke(gpu::declval<_Proj&>(), gpu::declval<_Tp>()))
  >::type
  _LIBGPU_CONSTEXPR operator()(_Tp&& __v) const {
    return std::__invoke(__pred, std::__invoke(__proj, std::forward<_Tp>(__v)));
  }

  template <class _T1, class _T2>
  typename __invoke_of<_Pred&,
                       decltype(std::__invoke(gpu::declval<_Proj&>(), gpu::declval<_T1>())),
                       decltype(std::__invoke(gpu::declval<_Proj&>(), gpu::declval<_T2>()))
  >::type
  _LIBGPU_CONSTEXPR operator()(_T1&& __lhs, _T2&& __rhs) const {
    return std::__invoke(__pred,
                      std::__invoke(__proj, std::forward<_T1>(__lhs)),
                      std::__invoke(__proj, std::forward<_T2>(__rhs)));
  }

};

template <class _Pred,
          class _Proj,
          std::enable_if_t<!(!std::is_member_pointer<std::decay_t<_Pred> >::value &&
                            __is_identity<std::decay_t<_Proj> >::value),
                        int> = 0>
__device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR _ProjectedPred<_Pred, _Proj>
__make_projected(_Pred& __pred, _Proj& __proj) {
  return _ProjectedPred<_Pred, _Proj>(__pred, __proj);
}

// Avoid creating the functor and just use the pristine comparator -- for certain algorithms, this would enable
// optimizations that rely on the type of the comparator. Additionally, this results in less layers of indirection in
// the call stack when the comparator is invoked, even in an unoptimized build.
template <class _Pred,
          class _Proj,
          std::enable_if_t<!std::is_member_pointer<std::decay_t<_Pred> >::value &&
                          __is_identity<std::decay_t<_Proj> >::value,
                        int> = 0>
__device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR _Pred& __make_projected(_Pred& __pred, _Proj&) {
  return __pred;
}

_LIBGPU_END_NAMESPACE_STD

#if _LIBGPU_STD_VER >= 20

_LIBGPU_BEGIN_NAMESPACE_STD

namespace ranges {

template <class _Comp, class _Proj1, class _Proj2>
__device__ _LIBGPU_HIDE_FROM_ABI constexpr
decltype(auto) __make_projected_comp(_Comp& __comp, _Proj1& __proj1, _Proj2& __proj2) {
  if constexpr (__is_identity<std::decay_t<_Proj1>>::value && __is_identity<std::decay_t<_Proj2>>::value &&
                !std::is_member_pointer_v<std::decay_t<_Comp>>) {
    // Avoid creating the lambda and just use the pristine comparator -- for certain algorithms, this would enable
    // optimizations that rely on the type of the comparator.
    return __comp;

  } else {
    return [&](auto&& __lhs, auto&& __rhs) {
      return std::invoke(__comp,
                        std::invoke(__proj1, std::forward<decltype(__lhs)>(__lhs)),
                        std::invoke(__proj2, std::forward<decltype(__rhs)>(__rhs)));
    };
  }
}

} // namespace ranges

} // namespace gpu

#endif // __GPU___ALGORITHM_MAKE_PROJECTED_H

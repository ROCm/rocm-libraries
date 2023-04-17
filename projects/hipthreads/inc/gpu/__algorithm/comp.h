//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_COMP_H
#define __GPU___ALGORITHM_COMP_H

#include "gpu/__config"
#include "gpu/__type_traits/predicate_traits.h"

namespace gpu {

struct __equal_to {
  template <class _T1, class _T2>
  _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14 bool operator()(const _T1& __x, const _T2& __y) const {
    return __x == __y;
  }
};

template <class _Lhs, class _Rhs>
struct __is_trivial_equality_predicate<__equal_to, _Lhs, _Rhs> : std::true_type {};

template <class _T1, class _T2 = _T1>
struct __less
{
    _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX14
    bool operator()(const _T1& __x, const _T1& __y) const {return __x < __y;}

    _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX14
    bool operator()(const _T1& __x, const _T2& __y) const {return __x < __y;}

    _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX14
    bool operator()(const _T2& __x, const _T1& __y) const {return __x < __y;}

    _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX14
    bool operator()(const _T2& __x, const _T2& __y) const {return __x < __y;}
};

template <class _T1>
struct __less<_T1, _T1>
{
    _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX14
    bool operator()(const _T1& __x, const _T1& __y) const {return __x < __y;}
};

template <class _T1>
struct __less<const _T1, _T1>
{
    _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX14
    bool operator()(const _T1& __x, const _T1& __y) const {return __x < __y;}
};

template <class _T1>
struct __less<_T1, const _T1>
{
    _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX14
    bool operator()(const _T1& __x, const _T1& __y) const {return __x < __y;}
};

} // namespace gpu

#endif // __GPU___ALGORITHM_COMP_H

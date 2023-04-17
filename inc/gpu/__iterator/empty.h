// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ITERATOR_EMPTY_H
#define __GPU___ITERATOR_EMPTY_H

#include "gpu/__config"

namespace gpu {

#if _LIBGPU_STD_VER >= 17

template <class _Cont>
_LIBGPU_NODISCARD_AFTER_CXX17 _LIBGPU_INLINE_VISIBILITY
constexpr auto empty(const _Cont& __c)
_NOEXCEPT_(noexcept(__c.empty()))
-> decltype        (__c.empty())
{ return            __c.empty(); }

template <class _Tp, std::size_t _Sz>
_LIBGPU_NODISCARD_AFTER_CXX17 _LIBGPU_INLINE_VISIBILITY
constexpr bool empty(const _Tp (&)[_Sz]) noexcept { return false; }

template <class _Ep>
_LIBGPU_NODISCARD_AFTER_CXX17 _LIBGPU_INLINE_VISIBILITY
constexpr bool empty(std::initializer_list<_Ep> __il) noexcept { return __il.size() == 0; }

#endif // _LIBGPU_STD_VER >= 17

} // namespace gpu

#endif // __GPU___ITERATOR_EMPTY_H

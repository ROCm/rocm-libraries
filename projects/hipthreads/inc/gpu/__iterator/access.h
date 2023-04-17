// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ITERATOR_ACCESS_H
#define __GPU___ITERATOR_ACCESS_H

#include "gpu/__config"

namespace gpu {

template <class _Tp, std::size_t _Np>
_LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX14
_Tp*
begin(_Tp (&__array)[_Np])
{
    return __array;
}

template <class _Tp, std::size_t _Np>
_LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX14
_Tp*
end(_Tp (&__array)[_Np])
{
    return __array + _Np;
}

#if !defined(_LIBGPU_CXX03_LANG)

template <class _Cp>
_LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
auto
begin(_Cp& __c) -> decltype(__c.begin())
{
    return __c.begin();
}

template <class _Cp>
_LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
auto
begin(const _Cp& __c) -> decltype(__c.begin())
{
    return __c.begin();
}

template <class _Cp>
_LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
auto
end(_Cp& __c) -> decltype(__c.end())
{
    return __c.end();
}

template <class _Cp>
_LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
auto
end(const _Cp& __c) -> decltype(__c.end())
{
    return __c.end();
}

#if _LIBGPU_STD_VER >= 14

template <class _Cp>
_LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX14
auto cbegin(const _Cp& __c) -> decltype(std::begin(__c))
{
    return std::begin(__c);
}

template <class _Cp>
_LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX14
auto cend(const _Cp& __c) -> decltype(std::end(__c))
{
    return std::end(__c);
}

#endif


#else  // defined(_LIBGPU_CXX03_LANG)

template <class _Cp>
_LIBGPU_INLINE_VISIBILITY
typename _Cp::iterator
begin(_Cp& __c)
{
    return __c.begin();
}

template <class _Cp>
_LIBGPU_INLINE_VISIBILITY
typename _Cp::const_iterator
begin(const _Cp& __c)
{
    return __c.begin();
}

template <class _Cp>
_LIBGPU_INLINE_VISIBILITY
typename _Cp::iterator
end(_Cp& __c)
{
    return __c.end();
}

template <class _Cp>
_LIBGPU_INLINE_VISIBILITY
typename _Cp::const_iterator
end(const _Cp& __c)
{
    return __c.end();
}

#endif // !defined(_LIBGPU_CXX03_LANG)

} // namespace gpu

#endif // __GPU___ITERATOR_ACCESS_H

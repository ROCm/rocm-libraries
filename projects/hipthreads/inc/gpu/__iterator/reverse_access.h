// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ITERATOR_REVERSE_ACCESS_H
#define __GPU___ITERATOR_REVERSE_ACCESS_H

#include "gpu/__config"

namespace gpu {

#if _LIBGPU_STD_VER >= 14

template <class _Tp, std::size_t _Np>
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
reverse_iterator<_Tp*> rbegin(_Tp (&__array)[_Np])
{
    return reverse_iterator<_Tp*>(__array + _Np);
}

template <class _Tp, std::size_t _Np>
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
reverse_iterator<_Tp*> rend(_Tp (&__array)[_Np])
{
    return reverse_iterator<_Tp*>(__array);
}

template <class _Ep>
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
reverse_iterator<const _Ep*> rbegin(std::initializer_list<_Ep> __il)
{
    return reverse_iterator<const _Ep*>(__il.end());
}

template <class _Ep>
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
reverse_iterator<const _Ep*> rend(std::initializer_list<_Ep> __il)
{
    return reverse_iterator<const _Ep*>(__il.begin());
}

template <class _Cp>
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
auto rbegin(_Cp& __c) -> decltype(__c.rbegin())
{
    return __c.rbegin();
}

template <class _Cp>
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
auto rbegin(const _Cp& __c) -> decltype(__c.rbegin())
{
    return __c.rbegin();
}

template <class _Cp>
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
auto rend(_Cp& __c) -> decltype(__c.rend())
{
    return __c.rend();
}

template <class _Cp>
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
auto rend(const _Cp& __c) -> decltype(__c.rend())
{
    return __c.rend();
}

template <class _Cp>
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
auto crbegin(const _Cp& __c) -> decltype(gpu::rbegin(__c))
{
    return gpu::rbegin(__c);
}

template <class _Cp>
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX17
auto crend(const _Cp& __c) -> decltype(gpu::rend(__c))
{
    return gpu::rend(__c);
}

#endif // _LIBGPU_STD_VER >= 14

} // namespace gpu

#endif // __GPU___ITERATOR_REVERSE_ACCESS_H

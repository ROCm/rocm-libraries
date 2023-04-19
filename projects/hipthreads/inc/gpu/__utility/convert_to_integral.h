//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___UTILITY_CONVERT_TO_INTEGRAL_H__
#define __GPU___UTILITY_CONVERT_TO_INTEGRAL_H__

#include "gpu/__config"
#include <type_traits>

namespace gpu {

//====================================================================================================================//
//      Adapted from libc++ __utility/convert_to_integral.h
//====================================================================================================================//

__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR
int __convert_to_integral(int __val) { return __val; }

__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR
unsigned __convert_to_integral(unsigned __val) { return __val; }

__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR
long __convert_to_integral(long __val) { return __val; }

__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR
unsigned long __convert_to_integral(unsigned long __val) { return __val; }

__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR
long long __convert_to_integral(long long __val) { return __val; }

__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR
unsigned long long __convert_to_integral(unsigned long long __val) {return __val; }

template<typename _Fp>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR
typename std::enable_if<std::is_floating_point<_Fp>::value, long long>::type
 __convert_to_integral(_Fp __val) { return __val; }

#ifndef _LIBGPU_HAS_NO_INT128
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR
__int128_t __convert_to_integral(__int128_t __val) { return __val; }

__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR
__uint128_t __convert_to_integral(__uint128_t __val) { return __val; }
#endif

template <class _Tp, bool = std::is_enum<_Tp>::value>
struct __sfinae_underlying_type
{
    typedef typename std::underlying_type<_Tp>::type type;
    typedef decltype((static_cast<type>(1)) + 0) __promoted_type;
};

template <class _Tp>
struct __sfinae_underlying_type<_Tp, false> {};

template <class _Tp>
__device__ inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR
typename __sfinae_underlying_type<_Tp>::__promoted_type
__convert_to_integral(_Tp __val) { return __val; }

} // namespace gpu

#endif // __GPU___UTILITY_CONVERT_TO_INTEGRAL_H__


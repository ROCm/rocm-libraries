// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___NUMERIC_MIDPOINT_H
#define __GPU___NUMERIC_MIDPOINT_H

#include "gpu/__config"

namespace gpu {

#if _LIBGPU_STD_VER >= 20
template <class _Tp>
__device__ _LIBGPU_INLINE_VISIBILITY constexpr
std::enable_if_t<std::is_integral_v<_Tp> && !std::is_same_v<bool, _Tp> && !std::is_null_pointer_v<_Tp>, _Tp>
midpoint(_Tp __a, _Tp __b) noexcept
_LIBGPU_DISABLE_UBSAN_UNSIGNED_INTEGER_CHECK
{
    using _Up = std::make_unsigned_t<_Tp>;
    constexpr _Up __bitshift = std::numeric_limits<_Up>::digits - 1;

    _Up __diff = _Up(__b) - _Up(__a);
    _Up __sign_bit = __b < __a;

    _Up __half_diff = (__diff / 2) + (__sign_bit << __bitshift) + (__sign_bit & __diff);

    return __a + __half_diff;
}


template <class _TPtr>
__device__ _LIBGPU_INLINE_VISIBILITY constexpr
std::enable_if_t<std::is_pointer_v<_TPtr>
             && std::is_object_v<std::remove_pointer_t<_TPtr>>
             && ! std::is_void_v<std::remove_pointer_t<_TPtr>>
             && (sizeof(std::remove_pointer_t<_TPtr>) > 0), _TPtr>
midpoint(_TPtr __a, _TPtr __b) noexcept
{
    return __a + gpu::midpoint(std::ptrdiff_t(0), __b - __a);
}


template <typename _Tp>
__device__ _LIBGPU_HIDE_FROM_ABI constexpr int __sign(_Tp __val) {
    return (_Tp(0) < __val) - (__val < _Tp(0));
}

template <typename _Fp>
__device__ _LIBGPU_HIDE_FROM_ABI constexpr _Fp __fp_abs(_Fp __f) { return __f >= 0 ? __f : -__f; }

template <class _Fp>
__device__ _LIBGPU_INLINE_VISIBILITY constexpr
std::enable_if_t<std::is_floating_point_v<_Fp>, _Fp>
midpoint(_Fp __a, _Fp __b) noexcept
{
    constexpr _Fp __lo = std::numeric_limits<_Fp>::min()*2;
    constexpr _Fp __hi = std::numeric_limits<_Fp>::max()/2;
    return std::__fp_abs(__a) <= __hi && std::__fp_abs(__b) <= __hi ?  // typical case: overflow is impossible
      (__a + __b)/2 :                                        // always correctly rounded
      std::__fp_abs(__a) < __lo ? __a + __b/2 :                   // not safe to halve a
      std::__fp_abs(__b) < __lo ? __a/2 + __b :                   // not safe to halve b
      __a/2 + __b/2;                                         // otherwise correctly rounded
}

#endif // _LIBGPU_STD_VER >= 20

} // namespace gpu

#endif // __GPU___NUMERIC_MIDPOINT_H

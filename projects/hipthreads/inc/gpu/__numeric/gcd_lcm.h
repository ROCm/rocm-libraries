// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___NUMERIC_GCD_LCM_H
#define __GPU___NUMERIC_GCD_LCM_H

#include "gpu/__config"

namespace gpu {

#if _LIBGPU_STD_VER >= 17

template <typename _Result, typename _Source, bool _IsSigned = std::is_signed<_Source>::value> struct __ct_abs;

template <typename _Result, typename _Source>
struct __ct_abs<_Result, _Source, true> {
    __device__ _LIBGPU_CONSTEXPR _LIBGPU_INLINE_VISIBILITY
    _Result operator()(_Source __t) const noexcept
    {
        if (__t >= 0) return __t;
        if (__t == numeric_limits<_Source>::min()) return -static_cast<_Result>(__t);
        return -__t;
    }
};

template <typename _Result, typename _Source>
struct __ct_abs<_Result, _Source, false> {
    __device__ _LIBGPU_CONSTEXPR _LIBGPU_INLINE_VISIBILITY
    _Result operator()(_Source __t) const noexcept { return __t; }
};


template<class _Tp>
_LIBGPU_CONSTEXPR _LIBGPU_HIDDEN
_Tp __gcd(_Tp __m, _Tp __n)
{
    static_assert((!std::is_signed<_Tp>::value), "");
    return __n == 0 ? __m : std::__gcd<_Tp>(__n, __m % __n);
}

template<class _Tp, class _Up>
__device__ _LIBGPU_CONSTEXPR _LIBGPU_INLINE_VISIBILITY
std::common_type_t<_Tp,_Up>
gcd(_Tp __m, _Up __n)
{
    static_assert((std::is_integral<_Tp>::value && std::is_integral<_Up>::value), "Arguments to gcd must be integer types");
    static_assert((!std::is_same<std::remove_cv_t<_Tp>, bool>::value), "First argument to gcd cannot be bool" );
    static_assert((!std::is_same<std::remove_cv_t<_Up>, bool>::value), "Second argument to gcd cannot be bool" );
    using _Rp = std::common_type_t<_Tp,_Up>;
    using _Wp = std::make_unsigned_t<_Rp>;
    return static_cast<_Rp>(std::__gcd(
        static_cast<_Wp>(__ct_abs<_Rp, _Tp>()(__m)),
        static_cast<_Wp>(__ct_abs<_Rp, _Up>()(__n))));
}

template<class _Tp, class _Up>
__device__ _LIBGPU_CONSTEXPR _LIBGPU_INLINE_VISIBILITY
std::common_type_t<_Tp,_Up>
lcm(_Tp __m, _Up __n)
{
    static_assert((std::is_integral<_Tp>::value && std::is_integral<_Up>::value), "Arguments to lcm must be integer types");
    static_assert((!std::is_same<std::remove_cv_t<_Tp>, bool>::value), "First argument to lcm cannot be bool" );
    static_assert((!std::is_same<std::remove_cv_t<_Up>, bool>::value), "Second argument to lcm cannot be bool" );
    if (__m == 0 || __n == 0)
        return 0;

    using _Rp = std::common_type_t<_Tp,_Up>;
    _Rp __val1 = __ct_abs<_Rp, _Tp>()(__m) / std::gcd(__m, __n);
    _Rp __val2 = __ct_abs<_Rp, _Up>()(__n);
    assert((numeric_limits<_Rp>::max() / __val1 > __val2) && "Overflow in lcm");
    return __val1 * __val2;
}

#endif // _LIBGPU_STD_VER

} // namespace gpu

#endif // __GPU___NUMERIC_GCD_LCM_H

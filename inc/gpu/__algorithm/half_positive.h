//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_HALF_POSITIVE_H
#define __GPU___ALGORITHM_HALF_POSITIVE_H

#include "gpu/__config"

namespace gpu {

// Perform division by two quickly for positive integers (llvm.org/PR39129)

template <typename _Integral>
_LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR
typename std::enable_if
<
    std::is_integral<_Integral>::value,
    _Integral
>::type
__half_positive(_Integral __value)
{
    return static_cast<_Integral>(static_cast<std::make_unsigned_t<_Integral> >(__value) / 2);
}

template <typename _Tp>
_LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR
typename std::enable_if
<
    !std::is_integral<_Tp>::value,
    _Tp
>::type
__half_positive(_Tp __value)
{
    return __value / 2;
}

} // namespace gpu

#endif // __GPU___ALGORITHM_HALF_POSITIVE_H

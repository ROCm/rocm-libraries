//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___TYPE_TRAITS_IS_TRIVIALLY_RELOCATABLE_H
#define __GPU___TYPE_TRAITS_IS_TRIVIALLY_RELOCATABLE_H

#include "gpu/__config"
#include <type_traits>

namespace gpu {

// A type is trivially relocatable if a move construct + destroy of the original object is equivalent to
// `memcpy(dst, src, sizeof(T))`.

#if __has_builtin(__is_trivially_relocatable)
template <class _Tp, class = void>
struct __libcpp_is_trivially_relocatable : std::integral_constant<bool, __is_trivially_relocatable(_Tp)> {};
#else
template <class _Tp, class = void>
struct __libcpp_is_trivially_relocatable : std::is_trivially_copyable<_Tp> {};
#endif

template <class _Tp>
struct __libcpp_is_trivially_relocatable<_Tp,
                                         std::enable_if_t<std::is_same<_Tp, typename _Tp::__trivially_relocatable>::value> >
    : std::true_type {};

} // namespace gpu

#endif // __GPU___TYPE_TRAITS_IS_TRIVIALLY_RELOCATABLE_H

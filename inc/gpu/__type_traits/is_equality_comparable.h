//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___TYPE_TRAITS_IS_EQUALITY_COMPARABLE_H__
#define __GPU___TYPE_TRAITS_IS_EQUALITY_COMPARABLE_H__

#include "gpu/__config"

#include <type_traits>

#include "gpu/__utility/declval.h"

namespace gpu {

//====================================================================================================================//
//      Adapted from libc++ __type_traits/is_equality_comparable.h
//====================================================================================================================//

template <class _Tp, class _Up, class = void>
struct __is_equality_comparable : std::false_type {};

template <class _Tp, class _Up>
struct __is_equality_comparable<_Tp, _Up, std::void_t<decltype(gpu::declval<_Tp>() == gpu::declval<_Up>())> > : std::true_type {
};

// A type is_trivially_equality_comparable if the expression `a == b` is equivalent to `std::memcmp(&a, &b, sizeof(T))`
// (with `a` and `b` being of type `T`). There is no compiler built-in to check this, so we can only do this for known
// types. In particular, these are the integral types and raw pointers.
//
// The following types are not trivially equality comparable:
// floating-point types: different bit-patterns can compare equal. (e.g 0.0 and -0.0)
// enums: The user is allowed to specialize operator== for enums
// pointers that don't have the same type (ignoring cv-qualifiers): pointers to virtual bases are equality comparable,
//   but don't have the same bit-pattern. An exception to this is comparing to a void-pointer. There the bit-pattern is
//   always compared.

template <class _Tp, class _Up>
struct __libcpp_is_trivially_equality_comparable
    : std::integral_constant<bool,
                        __is_equality_comparable<_Tp, _Up>::value && std::is_integral<_Tp>::value &&
                            std::is_same<std::remove_cv_t<_Tp>, std::remove_cv_t<_Up> >::value> {};

// TODO: Use is_pointer_inverconvertible_base_of
template <class _Tp, class _Up>
struct __libcpp_is_trivially_equality_comparable<_Tp*, _Up*>
    : std::integral_constant<
          bool,
          __is_equality_comparable<_Tp*, _Up*>::value &&
              (std::is_same<std::remove_cv_t<_Tp>, std::remove_cv_t<_Up> >::value || std::is_void<_Tp>::value || std::is_void<_Up>::value)> {
};

} // namespace gpu

#endif // __GPU___TYPE_TRAITS_IS_EQUALITY_COMPARABLE_H__

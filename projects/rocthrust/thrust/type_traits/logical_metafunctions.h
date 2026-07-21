/*
 *  Copyright 2008-2021 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _THRUST_HAS_DEVICE_SYSTEM_STD
// clang-format off
#  include _THRUST_STD_INCLUDE(__type_traits/conjunction.h)
#  include _THRUST_STD_INCLUDE(__type_traits/disjunction.h)
#  include _THRUST_STD_INCLUDE(__type_traits/negation.h)
// clang-format on
#else
#  include <type_traits>
#endif

THRUST_NAMESPACE_BEGIN

using _THRUST_STD::conjunction;
using _THRUST_STD::conjunction_v;
using _THRUST_STD::disjunction;
using _THRUST_STD::disjunction_v;
using _THRUST_STD::negation;
using _THRUST_STD::negation_v;

template <bool... Bs>
using conjunction_value THRUST_DEPRECATED_BECAUSE("Use: _THRUST_STD::bool_constant<(Bs && ...)>") =
  conjunction<_THRUST_STD::bool_constant<Bs>...>;

template <bool... Bs>
using disjunction_value THRUST_DEPRECATED_BECAUSE("Use: _THRUST_STD::bool_constant<(Bs || ...)>") =
  disjunction<_THRUST_STD::bool_constant<Bs>...>;

template <bool B>
using negation_value THRUST_DEPRECATED_BECAUSE("Use _THRUST_STD::bool_constant<!B>") = _THRUST_STD::bool_constant<!B>;

THRUST_SUPPRESS_DEPRECATED_PUSH
template <bool... Bs>
constexpr bool
  conjunction_value_v THRUST_DEPRECATED_BECAUSE("Use a fold expression: Bs && ...") = conjunction_value<Bs...>::value;

template <bool... Bs>
constexpr bool
  disjunction_value_v THRUST_DEPRECATED_BECAUSE("Use a fold expression: Bs || ...") = disjunction_value<Bs...>::value;

template <bool B>
constexpr bool negation_value_v THRUST_DEPRECATED_BECAUSE("Use a plain negation !B") = negation_value<B>::value;
THRUST_SUPPRESS_DEPRECATED_POP

THRUST_NAMESPACE_END

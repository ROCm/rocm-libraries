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

#include <thrust/detail/type_traits.h>

THRUST_NAMESPACE_BEGIN

#ifndef THRUST_DOXYGEN_INVOKED
using ::internal::conjunction;
using ::internal::conjunction_v;
using ::internal::disjunction;
using ::internal::disjunction_v;
using ::internal::negation;
using ::internal::negation_v;
#endif

template <bool... Bs>
using conjunction_value THRUST_DEPRECATED_BECAUSE("Use: internal::bool_constant<(Bs && ...)>") =
  conjunction<::internal::bool_constant<Bs>...>;

template <bool... Bs>
constexpr bool
  conjunction_value_v THRUST_DEPRECATED_BECAUSE("Use a fold expression: Bs && ...") = conjunction_value<Bs...>::value;

template <bool... Bs>
using disjunction_value THRUST_DEPRECATED_BECAUSE("Use: internal::bool_constant<(Bs || ...)>") =
  disjunction<::internal::bool_constant<Bs>...>;

template <bool... Bs>
constexpr bool
  disjunction_value_v THRUST_DEPRECATED_BECAUSE("Use a fold expression: Bs || ...") = disjunction_value<Bs...>::value;

template <bool B>
using negation_value THRUST_DEPRECATED_BECAUSE("Use internal::bool_constant<!B>") = ::internal::bool_constant<!B>;

template <bool B>
constexpr bool negation_value_v THRUST_DEPRECATED_BECAUSE("Use a plain negation !B") = negation_value<B>::value;

THRUST_NAMESPACE_END

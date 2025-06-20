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

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#  include <cuda/std/__type_traits/conjunction.h>
#  include <cuda/std/__type_traits/disjunction.h>
#  include <cuda/std/__type_traits/negation.h>
#else
#  include <type_traits>
#endif

THRUST_NAMESPACE_BEGIN

#ifndef THRUST_DOXYGEN_INVOKED
#  if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
using ::cuda::std::conjunction;
using ::cuda::std::conjunction_v;
using ::cuda::std::disjunction;
using ::cuda::std::disjunction_v;
using ::cuda::std::negation;
using ::cuda::std::negation_v;
#  else
using ::std::conjunction;
using ::std::conjunction_v;
using ::std::disjunction;
using ::std::disjunction_v;
using ::std::negation;
using ::std::negation_v;
#  endif
#endif

template <bool... Bs>
using conjunction_value THRUST_DEPRECATED_BECAUSE("Use: cuda::std::bool_constant<(Bs && ...)>") =
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
  conjunction<::cuda::std::bool_constant<Bs>...>;
#else
  conjunction<::std::bool_constant<Bs>...>;
#endif

template <bool... Bs>
constexpr bool
  conjunction_value_v THRUST_DEPRECATED_BECAUSE("Use a fold expression: Bs && ...") = conjunction_value<Bs...>::value;

template <bool... Bs>
using disjunction_value THRUST_DEPRECATED_BECAUSE("Use: cuda::std::bool_constant<(Bs || ...)>") =
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
  disjunction<::cuda::std::bool_constant<Bs>...>;
#else
  disjunction<::std::bool_constant<Bs>...>;
#endif

template <bool... Bs>
constexpr bool
  disjunction_value_v THRUST_DEPRECATED_BECAUSE("Use a fold expression: Bs || ...") = disjunction_value<Bs...>::value;

template <bool B>
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
using negation_value THRUST_DEPRECATED_BECAUSE("Use cuda::std::bool_constant<!B>") = ::cuda::std::bool_constant<!B>;
#else
using negation_value THRUST_DEPRECATED_BECAUSE("Use cuda::std::bool_constant<!B>") = ::std::bool_constant<!B>;
#endif

template <bool B>
constexpr bool negation_value_v THRUST_DEPRECATED_BECAUSE("Use a plain negation !B") = negation_value<B>::value;

THRUST_NAMESPACE_END

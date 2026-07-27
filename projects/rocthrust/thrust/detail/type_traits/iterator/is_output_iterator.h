/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

#include <thrust/iterator/detail/any_assign.h>
#include <thrust/iterator/iterator_traits.h>

#include _THRUST_STD_INCLUDE(type_traits)

THRUST_NAMESPACE_BEGIN

namespace detail
{
template <typename T, typename SFINAE = void>
// TODO(libhipcxx): replace inline with _CCCL_INLINE_VAR once libhipcxx gets ready
inline constexpr bool is_output_iterator = true;

template <typename T>
// TODO(libhipcxx): replace inline with _CCCL_INLINE_VAR once libhipcxx gets ready
inline constexpr bool is_output_iterator<T, _THRUST_STD::void_t<it_value_t<T>>> =
  _THRUST_STD::is_void_v<it_value_t<T>> || _THRUST_STD::is_same_v<it_value_t<T>, any_assign>;
} // namespace detail

THRUST_NAMESPACE_END

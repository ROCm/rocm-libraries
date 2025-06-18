/*
 *  Copyright 2018-2021 NVIDIA Corporation
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

#if THRUST_HAS_INCLUDE(<version>)
#  include <version>
#endif // THRUST_HAS_INCLUDE(<version>)

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#  include <cuda/std/type_traits>
#else
#  include <type_traits>
#endif

THRUST_NAMESPACE_BEGIN

template <typename T>
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
using remove_cvref THRUST_DEPRECATED_BECAUSE("Use cuda::std::remove_cvref") = ::cuda::std::remove_cvref<T>;
#else
using remove_cvref
  THRUST_DEPRECATED_BECAUSE("Use cuda::std::remove_cvref") = ::std::remove_cv<::std::remove_reference_t<T>>;
#endif

template <typename T>
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
using remove_cvref_t THRUST_DEPRECATED_BECAUSE("Use cuda::std::remove_cvref_t") = ::cuda::std::remove_cvref_t<T>;
#else
using remove_cvref_t
  THRUST_DEPRECATED_BECAUSE("Use cuda::std::remove_cvref_t") = ::std::remove_cv_t<::std::remove_reference_t<T>>;
#endif

THRUST_NAMESPACE_END

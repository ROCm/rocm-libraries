/*
 *  Copyright 2008-2024 NVIDIA Corporation
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

#include <thrust/detail/execution_policy.h>
#include <thrust/detail/pointer.h>

THRUST_NAMESPACE_BEGIN

template <typename DerivedPolicy>
THRUST_HOST_DEVICE pointer<void, DerivedPolicy>
malloc(const thrust::detail::execution_policy_base<DerivedPolicy>& exec, std::size_t n);

template <typename T, typename DerivedPolicy>
THRUST_HOST_DEVICE pointer<T, DerivedPolicy>
malloc(const thrust::detail::execution_policy_base<DerivedPolicy>& exec, std::size_t n);

template <typename DerivedPolicy, typename Pointer>
THRUST_HOST_DEVICE void free(const thrust::detail::execution_policy_base<DerivedPolicy>& exec, Pointer ptr);

THRUST_NAMESPACE_END

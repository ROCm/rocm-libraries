/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
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

THRUST_NAMESPACE_BEGIN
namespace system::detail::sequential
{
struct tag;

template <typename>
struct execution_policy;

template <>
struct execution_policy<tag> : thrust::execution_policy<tag>
{
  using tag_type = tag;
};

struct tag : execution_policy<tag>
{
  constexpr tag() = default;

  // allow any execution_policy to convert to the sequential tag. required for minimum_system to pick the sequential one
  template <typename DerivedPolicy>
  THRUST_HOST_DEVICE tag(const thrust::execution_policy<DerivedPolicy>&)
  {}
};

template <typename Derived>
struct execution_policy : thrust::execution_policy<Derived>
{
  using tag_type = tag;

  // allow conversion to tag when it is not a successor
  THRUST_HOST_DEVICE operator tag() const
  {
    return {};
  }
};

// TODO(bgruber): do we need this global variable? We already have thrust::seq
THRUST_GLOBAL_CONSTANT tag seq;

} // namespace system::detail::sequential
THRUST_NAMESPACE_END

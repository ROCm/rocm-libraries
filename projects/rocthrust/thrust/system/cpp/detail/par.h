/*
 *  Copyright 2008-2018 NVIDIA Corporation
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

#include <thrust/detail/allocator_aware_execution_policy.h>
#include <thrust/system/cpp/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace cpp
{
namespace detail
{

struct par_t
    : thrust::system::cpp::detail::execution_policy<par_t>
    , thrust::detail::allocator_aware_execution_policy<thrust::system::cpp::detail::execution_policy>
{
  THRUST_HOST_DEVICE constexpr par_t()
      : thrust::system::cpp::detail::execution_policy<par_t>()
  {}
};

} // namespace detail

THRUST_INLINE_CONSTANT detail::par_t par;

} // namespace cpp
} // namespace system

// alias par here
namespace cpp
{

using thrust::system::cpp::par;

} // namespace cpp
THRUST_NAMESPACE_END

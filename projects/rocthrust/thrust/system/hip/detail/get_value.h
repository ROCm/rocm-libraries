/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HIP
#  include <thrust/system/hip/config.h>

#  include <thrust/detail/raw_pointer_cast.h>
#  include <thrust/iterator/iterator_traits.h>
#  include <thrust/system/detail/adl/assign_value.h>
#  include <thrust/system/hip/detail/cross_system.h>
#  include <thrust/system/hip/detail/nv/target.h>

THRUST_NAMESPACE_BEGIN
namespace hip_rocprim
{

template <typename DerivedPolicy, typename Pointer>
typename thrust::iterator_value<Pointer>::type THRUST_HIP_FUNCTION
get_value(execution_policy<DerivedPolicy>& exec, Pointer ptr)
{
  using result_type = typename thrust::iterator_value<Pointer>::type;

  // WORKAROUND
  NV_IF_TARGET(
    NV_IS_HOST,
    (
      // when called from host code, implement with assign_value
      // note that this requires a type with default constructor
      result_type result;

      thrust::host_system_tag host_tag;
      cross_system<thrust::host_system_tag, DerivedPolicy> systems(host_tag, exec);
      assign_value(systems, &result, ptr);

      return result;),
    (THRUST_UNUSED_VAR(exec);
     void (*fptr)(cross_system<thrust::host_system_tag, DerivedPolicy>&, result_type*, Pointer) = assign_value;
     (void) fptr;

     return *thrust::raw_pointer_cast(ptr);))
} // end get_value()

} // namespace hip_rocprim
THRUST_NAMESPACE_END

#endif

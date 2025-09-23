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

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HIP
#  include <thrust/system/hip/config.h>

#  include <thrust/detail/raw_pointer_cast.h>
#  include <thrust/system/hip/detail/copy.h>
#  include <thrust/system/hip/detail/execution_policy.h>
#  include <thrust/system/hip/detail/nv/target.h>

THRUST_NAMESPACE_BEGIN
namespace hip_rocprim
{

template <typename DerivedPolicy, typename Pointer1, typename Pointer2>
inline THRUST_HOST_DEVICE void
assign_value(thrust::hip::execution_policy<DerivedPolicy>& exec, Pointer1 dst, Pointer2 src)
{
  // XXX war nvbugs/881631
  struct war_nvbugs_881631
  {
    THRUST_HOST inline static void
    host_path(thrust::hip::execution_policy<DerivedPolicy>& exec, Pointer1 dst, Pointer2 src)
    {
      hip_rocprim::copy(exec, src, src + 1, dst);
    }

    THRUST_DEVICE inline static void
    device_path(thrust::hip::execution_policy<DerivedPolicy>&, Pointer1 dst, Pointer2 src)
    {
      *thrust::raw_pointer_cast(dst) = *thrust::raw_pointer_cast(src);
    }
  };

  NV_IF_TARGET(
    NV_IS_HOST, (war_nvbugs_881631::host_path(exec, dst, src);), (war_nvbugs_881631::device_path(exec, dst, src);));

} // end assign_value()

template <typename System1, typename System2, typename Pointer1, typename Pointer2>
inline THRUST_HOST_DEVICE void assign_value(cross_system<System1, System2>& systems, Pointer1 dst, Pointer2 src)
{
  // XXX war nvbugs/881631
  struct war_nvbugs_881631
  {
    THRUST_HOST inline static void host_path(cross_system<System1, System2>& systems, Pointer1 dst, Pointer2 src)
    {
      // rotate the systems so that they are ordered the same as (src, dst)
      // for the call to thrust::copy
      cross_system<System2, System1> rotated_systems = systems.rotate();
      hip_rocprim::copy(rotated_systems, src, src + 1, dst);
    }

    THRUST_DEVICE inline static void device_path(cross_system<System1, System2>&, Pointer1 dst, Pointer2 src)
    {
      // XXX forward the true hip::execution_policy inside systems here
      //     instead of materializing a tag
      thrust::hip::tag hip_tag;
      thrust::hip_rocprim::assign_value(hip_tag, dst, src);
    }
  };

  NV_IF_TARGET(NV_IS_HOST,
               (war_nvbugs_881631::host_path(systems, dst, src);),
               (war_nvbugs_881631::device_path(systems, dst, src);));
} // end assign_value()

} // namespace hip_rocprim
THRUST_NAMESPACE_END
#endif

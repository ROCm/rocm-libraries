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

#if THRUST_HAS_HIP_COMPILER()
#  include <thrust/system/hip/config.h>

#  include <thrust/detail/raw_pointer_cast.h>
#  include <thrust/system/hip/detail/copy.h>
#  include <thrust/system/hip/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace hip_rocprim
{
template <typename DerivedPolicy, typename Pointer1, typename Pointer2>
THRUST_HOST_DEVICE void assign_value([[maybe_unused]] execution_policy<DerivedPolicy>& exec, Pointer1 dst, Pointer2 src)
{
  // Because of https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-arch point 2., if a call from a __host__
  // __device__ function leads to the template instantiation of a __global__ function, then this instantiation needs to
  // happen regardless of whether __CUDA_ARCH__ is defined. Therefore, we make the host path visible outside the
  // _THRUST_IF_TARGET switch. See also NVBug 881631.
  struct HostPath
  {
    THRUST_HOST auto operator()(execution_policy<DerivedPolicy>& exec, Pointer1 dst, Pointer2 src)
    {
      hip_rocprim::copy(exec, src, src + 1, dst);
    }
  };
  _THRUST_IF_TARGET(
    _THRUST_IS_HOST, (HostPath{}(exec, dst, src);), *thrust::raw_pointer_cast(dst) = *thrust::raw_pointer_cast(src););
}

template <typename System1, typename System2, typename Pointer1, typename Pointer2>
THRUST_HOST_DEVICE void
assign_value([[maybe_unused]] cross_system<System1, System2>& systems, Pointer1 dst, Pointer2 src)
{
  // Because of https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-arch point 2., if a call from a __host__
  // __device__ function leads to the template instantiation of a __global__ function, then this instantiation needs to
  // happen regardless of whether __CUDA_ARCH__ is defined. Therefore, we make the host path visible outside the
  // _THRUST_IF_TARGET switch. See also NVBug 881631.
  struct HostPath
  {
    THRUST_HOST auto operator()(cross_system<System1, System2>& systems, Pointer1 dst, Pointer2 src)
    {
      // rotate the systems so that they are ordered the same as (src, dst) for the call to thrust::copy
      cross_system<System2, System1> rotated_systems = systems.rotate();
      hip_rocprim::copy(rotated_systems, src, src + 1, dst);
    }
  };
  _THRUST_IF_TARGET(
    _THRUST_IS_HOST,
    (HostPath{}(systems, dst, src);),
    (
      // XXX forward the true hip::execution_policy inside systems here instead of materializing a tag
      hip::tag hip_tag; hip_rocprim::assign_value(hip_tag, dst, src);));
}
} // namespace hip_rocprim
THRUST_NAMESPACE_END
#endif

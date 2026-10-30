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

#if THRUST_HAS_HIP_COMPILER()

#  include <thrust/system/hip/config.h>

#  include <thrust/detail/raw_pointer_cast.h>
#  include <thrust/system/hip/detail/execution_policy.h>
#  include <thrust/system/hip/detail/util.h>

#  if _THRUST_HAS_DEVICE_SYSTEM_STD
#    include _THRUST_STD_INCLUDE(__utility/swap.h)
#  else
#    include <thrust/swap.h>
#  endif

THRUST_NAMESPACE_BEGIN
namespace hip_rocprim
{
namespace detail
{
template <typename Pointer1, typename Pointer2>
THRUST_DEVICE inline void iter_swap_device(Pointer1 a, Pointer2 b)
{
#  if _THRUST_HAS_DEVICE_SYSTEM_STD
  using _THRUST_STD::swap;
#  else
  using thrust::swap;
#  endif
  swap(*thrust::raw_pointer_cast(a), *thrust::raw_pointer_cast(b));
}

template <typename Pointer1, typename Pointer2>
ROCPRIM_KERNEL void iter_swap_kernel(Pointer1 a, Pointer2 b)
{
  detail::iter_swap_device(a, b);
}
} // namespace detail

template <typename DerivedPolicy, typename Pointer1, typename Pointer2>
inline THRUST_HOST_DEVICE void iter_swap(thrust::hip::execution_policy<DerivedPolicy>& exec, Pointer1 a, Pointer2 b)
{
  // Because of https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-arch point 2., if a call from a __host__
  // __device__ function leads to the template instantiation of a __global__ function, then this instantiation needs to
  // happen regardless of whether __CUDA_ARCH__ is defined. Therefore, we make the host path visible outside the
  // _THRUST_IF_TARGET switch. See also NVBug 881631.
  struct host_path
  {
    THRUST_HOST void operator()(execution_policy<DerivedPolicy>& exec, Pointer1 a, Pointer2 b) const
    {
      hipLaunchKernelGGL(
        HIP_KERNEL_NAME(detail::iter_swap_kernel<Pointer1, Pointer2>),
        dim3(1),
        dim3(1),
        0,
        hip_rocprim::stream(exec),
        a,
        b);
      hip_rocprim::throw_on_error(hipPeekAtLastError(), "iter_swap: calling iter_swap_kernel failed");
      hip_rocprim::throw_on_error(hip_rocprim::synchronize_optional(exec), "iter_swap: sync failed");
    }
  };

  _THRUST_IF_TARGET(_THRUST_IS_HOST, (host_path{}(exec, a, b);), (detail::iter_swap_device(a, b);));

} // end iter_swap()

} // namespace hip_rocprim
THRUST_NAMESPACE_END
#endif

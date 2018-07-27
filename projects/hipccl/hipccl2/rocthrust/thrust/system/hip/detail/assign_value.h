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

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
#include <thrust/detail/config.h>
#include <thrust/system/hip/config.h>
#include <thrust/system/hip/detail/execution_policy.h>
#include <thrust/detail/raw_pointer_cast.h>
// #include <thrust/system/hip/detail/copy.h>


BEGIN_NS_THRUST
namespace hip_rocprim {


template<typename DerivedPolicy, typename Pointer1, typename Pointer2>
inline __host__ __device__
  void assign_value(thrust::hip::execution_policy<DerivedPolicy> &exec, Pointer1 dst, Pointer2 src)
{
  // XXX war nvbugs/881631
  struct war_nvbugs_881631
  {
    __host__ inline static void host_path(thrust::hip::execution_policy<DerivedPolicy> &exec, Pointer1 dst, Pointer2 src)
    {
      (void) exec; (void) dst; (void) src;
      // hip_rocprim::copy(exec, src, src + 1, dst);
    }

    __device__ inline static void device_path(thrust::hip::execution_policy<DerivedPolicy> &, Pointer1 dst, Pointer2 src)
    {
      *thrust::raw_pointer_cast(dst) = *thrust::raw_pointer_cast(src);
    }
  };

#ifndef __HIP_DEVICE_COMPILE__
  war_nvbugs_881631::host_path(exec,dst,src);
#else
  war_nvbugs_881631::device_path(exec,dst,src);
#endif // __HIP_DEVICE_COMPILE__
} // end assign_value()


template<typename System1, typename System2, typename Pointer1, typename Pointer2>
inline __host__ __device__
  void assign_value(cross_system<System1,System2> &systems, Pointer1 dst, Pointer2 src)
{
  // XXX war nvbugs/881631
  struct war_nvbugs_881631
  {
    __host__ inline static void host_path(cross_system<System1,System2> &systems, Pointer1 dst, Pointer2 src)
    {
      // rotate the systems so that they are ordered the same as (src, dst)
      // for the call to thrust::copy
      cross_system<System2,System1> rotated_systems = systems.rotate();
      (void) rotated_systems; (void) dst; (void) src;
      // hip_rocprim::copy(rotated_systems, src, src + 1, dst);
    }

    __device__ inline static void device_path(cross_system<System1,System2> &, Pointer1 dst, Pointer2 src)
    {
      // XXX forward the true hip::execution_policy inside systems here
      //     instead of materializing a tag
      thrust::hip::tag hip_tag;
      thrust::hip_rocprim::assign_value(hip_tag, dst, src);
    }
  };

#if __HIP_DEVICE_COMPILE__
  war_nvbugs_881631::device_path(systems,dst,src);
#else
  war_nvbugs_881631::host_path(systems,dst,src);
#endif
} // end assign_value()




} // end hip_rocprim
END_NS_THRUST
#endif

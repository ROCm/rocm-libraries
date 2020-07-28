/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019 Advanced Micro Devices, Inc. All rights reserved.
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

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HIP
#include <thrust/system/hip/config.h>
#include <thrust/detail/config.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/system/hip/detail/copy.h>
#include <thrust/system/hip/detail/execution_policy.h>

THRUST_BEGIN_NS
namespace hip_rocprim
{

template <typename DerivedPolicy, typename Pointer1, typename Pointer2>
THRUST_HIP_FUNCTION void
assign_value(thrust::hip::execution_policy<DerivedPolicy>& exec, Pointer1 dst, Pointer2 src)
{
//WORKAROUND
#if defined(THRUST_HIP_DEVICE_CODE)

    THRUST_UNUSED_VAR(exec);
    Pointer1 (*fptr)(
        thrust::hip::execution_policy<DerivedPolicy>&, Pointer2, Pointer2, Pointer1)
        = hip_rocprim::copy;
    (void)fptr;

    *thrust::raw_pointer_cast(dst) = *thrust::raw_pointer_cast(src);
#else
    hip_rocprim::copy(exec, src, src + 1, dst);
#endif // __HIP_DEVICE_COMPILE__
} // end assign_value()

template <typename System1, typename System2, typename Pointer1, typename Pointer2>
THRUST_HIP_FUNCTION void
assign_value(cross_system<System1, System2>& systems, Pointer1 dst, Pointer2 src)
{
//WORKAROUND
#if defined(THRUST_HIP_DEVICE_CODE)
    THRUST_UNUSED_VAR(systems);
    Pointer1 (*fptr)(cross_system<System2, System1>, Pointer2, Pointer2, Pointer1)
        = hip_rocprim::copy;
    (void)fptr;
    // WORKAROUND build error fixed - start here
    // thrust::hip::tag hip_tag;
    // thrust::hip_rocprim::assign_value(hip_tag, dst, src);
    *thrust::raw_pointer_cast(dst) = *thrust::raw_pointer_cast(src);
    // WORKAROUND - end here
#else
    cross_system<System2, System1> rotated_systems = systems.rotate();
    hip_rocprim::copy(rotated_systems, src, src + 1, dst);
#endif

} // end assign_value()
} // end hip_rocprim
THRUST_END_NS
#endif

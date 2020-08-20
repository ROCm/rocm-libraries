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

#include <thrust/system/hip/detail/guarded_hip_runtime_api.h>

#include <thrust/detail/config.h>
#include <thrust/detail/seq.h>
#include <thrust/memory.h>
#include <thrust/system/hip/config.h>
// No caching allocator in rocPRIM
// #ifdef THRUST_CACHING_DEVICE_MALLOC
// #include <thrust/system/cuda/detail/cub/util_allocator.cuh>
// #endif
#include <thrust/system/detail/bad_alloc.h>
#include <thrust/system/hip/detail/util.h>

THRUST_BEGIN_NS
namespace hip_rocprim
{

// No caching allocator in rocPRIM
// #ifdef THRUST_CACHING_DEVICE_MALLOC
// #define __CUB_CACHING_MALLOC
// #ifndef __CUDA_ARCH__
// inline cub::CachingDeviceAllocator &get_allocator()
// {
//   static cub::CachingDeviceAllocator g_allocator(true);
//   return g_allocator;
// }
// #endif
// #endif

// note that malloc returns a raw pointer to avoid
// depending on the heavyweight thrust/system/hip/memory.h header
template <typename DerivedPolicy>
void* __host__ __device__
malloc(execution_policy<DerivedPolicy>&, std::size_t n)
{
    void* result = 0;

#ifndef __HIP_DEVICE_COMPILE__
    // No caching allocator in rocPRIM
    // #ifdef __CUB_CACHING_MALLOC
    //   cub::CachingDeviceAllocator &alloc = get_allocator();
    //   cudsError_t status = alloc.DeviceAllocate(&result, n);
    // #else
    hipError_t status = hipMalloc(&result, n);
    // #endif

    if(status != hipSuccess)
    {
        hipGetLastError(); // Clear global hip error state.
        throw thrust::system::detail::bad_alloc(thrust::hip_category().message(status).c_str());
    }
#else
    result = thrust::raw_pointer_cast(thrust::malloc(thrust::seq, n));
#endif

    return result;
} // end malloc()

template <typename DerivedPolicy, typename Pointer>
void __host__ __device__
free(execution_policy<DerivedPolicy>&, Pointer ptr)
{
#ifndef __HIP_DEVICE_COMPILE__
    // No caching allocator in rocPRIM
    // #ifdef __CUB_CACHING_MALLOC
    //   cub::CachingDeviceAllocator &alloc = get_allocator();
    //   hipError_t status = alloc.DeviceFree(thrust::raw_pointer_cast(ptr));
    // #else
    hipError_t status = hipFree(thrust::raw_pointer_cast(ptr));
    // #endif
    hip_rocprim::throw_on_error(status, "device free failed");
#else
    thrust::free(thrust::seq, ptr);
#endif
} // end free()

} // namespace hip_rocprim
THRUST_END_NS

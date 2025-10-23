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

#include <thrust/system/hip/config.h>

#include <thrust/detail/malloc_and_free.h>
#include <thrust/detail/malloc_and_free_fwd.h>
#include <thrust/detail/nv_target.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/detail/raw_reference_cast.h>
#include <thrust/detail/seq.h>
#include <thrust/system/detail/bad_alloc.h>
#include <thrust/system/hip/detail/util.h>

THRUST_NAMESPACE_BEGIN
namespace hip_rocprim
{

// note that malloc returns a raw pointer to avoid
// depending on the heavyweight thrust/system/hip/memory.h header
template <typename DerivedPolicy>
THRUST_HOST_DEVICE void* malloc(execution_policy<DerivedPolicy>&, std::size_t n)
{
  void* result = 0;

  NV_IF_TARGET(
    NV_IS_HOST,
    (hipError_t status = hipMalloc(&result, n);

     if (status != hipSuccess) {
       (void) hipGetLastError(); // Clear global HIP error state.
       throw thrust::system::detail::bad_alloc(thrust::hip_category().message(status).c_str());
     }),
    ( // NV_IS_DEVICE
      result = thrust::raw_pointer_cast(thrust::malloc(thrust::seq, n));));

  return result;
} // end malloc()

template <typename DerivedPolicy, typename Pointer>
THRUST_HOST_DEVICE void free(execution_policy<DerivedPolicy>&, Pointer ptr)
{
  NV_IF_TARGET(NV_IS_HOST,
               (hipError_t status = hipFree(thrust::raw_pointer_cast(ptr));
                hip_rocprim::throw_on_error(status, "device free failed");),
               ( // NV_IS_DEVICE
                 thrust::free(thrust::seq, ptr);));
} // end free()

} // namespace hip_rocprim
THRUST_NAMESPACE_END

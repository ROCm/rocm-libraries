/*
 *  Copyright 2008-2018 NVIDIA Corporation
 *  Modifications Copyright© 2018-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <thrust/system/hip/detail/malloc_and_free.h>
#include <thrust/system/hip/memory.h>

#include <limits>

THRUST_NAMESPACE_BEGIN
namespace hip_rocprim
{

THRUST_HOST_DEVICE pointer<void> malloc(std::size_t n)
{
  tag hip_tag;
  return pointer<void>(thrust::hip_rocprim::malloc(hip_tag, n));
} // end malloc()

template <typename T>
THRUST_HOST_DEVICE pointer<T> malloc(std::size_t n)
{
  pointer<void> raw_ptr = thrust::hip_rocprim::malloc(sizeof(T) * n);
  return pointer<T>(reinterpret_cast<T*>(raw_ptr.get()));
} // end malloc()

THRUST_HOST_DEVICE void free(pointer<void> ptr)
{
  tag hip_tag;
  return thrust::hip_rocprim::free(hip_tag, ptr.get());
} // end free()

} // namespace hip_rocprim
THRUST_NAMESPACE_END

/*
 *  Copyright 2008-2018 NVIDIA Corporation
 *  Modifications Copyright© 2018-2020 Advanced Micro Devices, Inc. All rights reserved.
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
#include <thrust/system/hip/detail/malloc_and_free.h>
#include <limits>

THRUST_NAMESPACE_BEGIN
namespace hip_rocprim
{

THRUST_HIP_FUNCTION pointer<void> malloc(std::size_t n)
{
    tag hip_tag;
    return pointer<void>(thrust::hip_rocprim::malloc(hip_tag, n));
} // end malloc()

template <typename T>
THRUST_HIP_FUNCTION pointer<T> malloc(std::size_t n)
{
    pointer<void> raw_ptr = thrust::hip_rocprim::malloc(sizeof(T) * n);
    return pointer<T>(reinterpret_cast<T*>(raw_ptr.get()));
} // end malloc()

THRUST_HIP_FUNCTION void free(pointer<void> ptr)
{
    tag hip_tag;
    return thrust::hip_rocprim::free(hip_tag, ptr.get());
} // end free()

} // end hip_rocprim
THRUST_NAMESPACE_END

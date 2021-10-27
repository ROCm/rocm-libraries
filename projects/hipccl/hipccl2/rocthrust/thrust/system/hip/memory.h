/*
 *  Copyright 2008-2018 NVIDIA Corporation
 * Modifications Copyright© 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
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

/*! \file thrust/system/hip/memory.h
 *  \brief Managing memory associated with Thrust's HIP system.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/hip/memory_resource.h>
#include <thrust/memory.h>
#include <thrust/detail/type_traits.h>
#include <thrust/mr/allocator.h>
#include <ostream>

THRUST_NAMESPACE_BEGIN
namespace hip_rocprim
{

/*! Allocates an area of memory available to Thrust's <tt>hip</tt> system.
 *  \param n Number of bytes to allocate.
 *  \return A <tt>hip::pointer<void></tt> pointing to the beginning of the newly
 *          allocated memory. A null <tt>hip::pointer<void></tt> is returned if
 *          an error occurs.
 *  \note The <tt>hip::pointer<void></tt> returned by this function must be
 *        deallocated with \p hip::free.
 *  \see hip::free
 *  \see std::malloc
 */
inline __host__ __device__ pointer<void> malloc(std::size_t n);

/*! Allocates a typed area of memory available to Thrust's <tt>hip</tt> system.
 *  \param n Number of elements to allocate.
 *  \return A <tt>hip::pointer<T></tt> pointing to the beginning of the newly
 *          allocated elements. A null <tt>hip::pointer<T></tt> is returned if
 *          an error occurs.
 *  \note The <tt>hip::pointer<T></tt> returned by this function must be
 *        deallocated with \p hip::free.
 *  \see hip::free
 *  \see std::malloc
 */
template <typename T>
inline __host__ __device__ pointer<T> malloc(std::size_t n);

/*! Deallocates an area of memory previously allocated by <tt>hip::malloc</tt>.
 *  \param ptr A <tt>hip::pointer<void></tt> pointing to the beginning of an area
 *         of memory previously allocated with <tt>hip::malloc</tt>.
 *  \see hip::malloc
 *  \see std::free
 */
inline __host__ __device__ void free(pointer<void> ptr);

/*! \p hip::allocator is the default allocator used by the \p hip system's
 *  containers such as <tt>hip::vector</tt> if no user-specified allocator is
 *  provided. \p hip::allocator allocates (deallocates) storage with \p
 *  hip::malloc (\p hip::free).
 */
template<typename T>
using allocator = thrust::mr::stateless_resource_allocator<
  T, thrust::system::hip::memory_resource
>;

/*! \p hip::universal_allocator allocates memory that can be used by the \p hip
 *  system and host systems.
 */
template<typename T>
using universal_allocator = thrust::mr::stateless_resource_allocator<
  T, thrust::system::hip::universal_memory_resource
>;

} // namespace hip_rocprim

namespace system { namespace hip
{
using thrust::hip_rocprim::malloc;
using thrust::hip_rocprim::free;
using thrust::hip_rocprim::allocator;
using thrust::hip_rocprim::universal_allocator;
}} // namespace system::hip

/*! \namespace thrust::hip
 *  \brief \p thrust::hip is a top-level alias for \p thrust::system::hip.
 */
namespace hip
{
using thrust::hip_rocprim::malloc;
using thrust::hip_rocprim::free;
using thrust::hip_rocprim::allocator;
using thrust::hip_rocprim::universal_allocator;
} // namespace hip

THRUST_NAMESPACE_END

#include <thrust/system/hip/detail/memory.inl>

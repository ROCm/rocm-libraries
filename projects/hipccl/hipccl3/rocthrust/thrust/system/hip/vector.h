/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright (c) 2019-2021, Advanced Micro Devices, Inc.  All rights reserved.
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

/*! \file thrust/system/hip/vector.h
 *  \brief A dynamically-sizable array of elements which reside in memory available to
 *         Thrust's HIP system.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/hip/memory.h>
#include <thrust/detail/vector_base.h>
#include <vector>

THRUST_NAMESPACE_BEGIN
namespace hip_rocprim
{

/*! \p hip::vector is a container that supports random access to elements,
 *  constant time removal of elements at the end, and linear time insertion
 *  and removal of elements at the beginning or in the middle. The number of
 *  elements in a \p hip::vector may vary dynamically; memory management is
 *  automatic. The elements contained in a \p hip::vector reside in memory
 *  accessible by the \p hip system.
 *
 *  \tparam T The element type of the \p hip::vector.
 *  \tparam Allocator The allocator type of the \p hip::vector.
 *          Defaults to \p hip::allocator.
 *
 *  \see https://en.cppreference.com/w/cpp/container/vector
 *  \see host_vector For the documentation of the complete interface which is
 *                   shared by \p hip::vector
 *  \see device_vector
 *  \see universal_vector
 */
template <typename T, typename Allocator = thrust::system::hip::allocator<T>>
using vector = thrust::detail::vector_base<T, Allocator>;

/*! \p hip::universal_vector is a container that supports random access to
 *  elements, constant time removal of elements at the end, and linear time
 *  insertion and removal of elements at the beginning or in the middle. The
 *  number of elements in a \p hip::universal_vector may vary dynamically;
 *  memory management is automatic. The elements contained in a
 *  \p hip::universal_vector reside in memory accessible by the \p hip system
 *  and host systems.
 *
 *  \tparam T The element type of the \p hip::universal_vector.
 *  \tparam Allocator The allocator type of the \p hip::universal_vector.
 *          Defaults to \p hip::universal_allocator.
 *
 *  \see https://en.cppreference.com/w/cpp/container/vector
 *  \see host_vector For the documentation of the complete interface which is
 *                   shared by \p hip::universal_vector
 *  \see device_vector
 *  \see universal_vector
 */
template <typename T, typename Allocator = thrust::system::hip::universal_allocator<T>>
using universal_vector = thrust::detail::vector_base<T, Allocator>;

} // namespace hip_rocprim

namespace system { namespace hip
{
using thrust::hip_rocprim::vector;
using thrust::hip_rocprim::universal_vector;
}}

namespace hip
{
using thrust::hip_rocprim::vector;
using thrust::hip_rocprim::universal_vector;
}

THRUST_NAMESPACE_END

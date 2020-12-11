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

/*! \file thrust/system/hip/vector.h
 *  \brief A dynamically-sizable array of elements which reside in memory available to
 *         Thrust's hip system.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/hip/memory.h>
#include <thrust/detail/vector_base.h>
#include <vector>

namespace thrust
{

// forward declaration of host_vector
template<typename T, typename Allocator> class host_vector;

namespace hip_rocprim
{

/*! \p hip_rocprim::vector is a container that supports random access to elements,
 *  constant time removal of elements at the end, and linear time insertion
 *  and removal of elements at the beginning or in the middle. The number of
 *  elements in a \p hip_rocprim::vector may vary dynamically; memory management is
 *  automatic. The elements contained in a \p hip_rocprim::vector reside in memory
 *  available to the \p hip_rocprim system.
 *
 *  \tparam T The element type of the \p hip_rocprim::vector.
 *  \tparam Allocator The allocator type of the \p hip_rocprim::vector. Defaults to \p hip_rocprim::allocator.
 *
 *  \see http://www.sgi.com/tech/stl/Vector.html
 *  \see host_vector For the documentation of the complete interface which is
 *                   shared by \p hip_rocprim::vector
 *  \see device_vector
 */
 template<typename T, typename Allocator = allocator<T> >
 using vector = thrust::detail::vector_base<T, Allocator>;


} // end hip_rocprim

// alias system::hip_rocprim names at top-level
namespace hip_rocprim
{

using thrust::hip_rocprim::vector;

} // end hip_rocprim

} // end thrust

#include <thrust/system/hip/detail/vector.inl>

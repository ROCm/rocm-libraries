/*
 *  Copyright 2008-2018 NVIDIA Corporation
 *  Modifications Copyright© 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
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
 *  \brief Managing memory associated with Thrust's hip system.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/hip/memory_resource.h>
#include <thrust/memory.h>
#include <thrust/detail/type_traits.h>
#include <thrust/mr/allocator.h>
#include <ostream>

THRUST_BEGIN_NS
namespace hip_rocprim
{

/*! \addtogroup system_backends Systems
 *  \ingroup system
 *  \{
 */

/*! \namespace thrust::system::hip
 *  \brief \p thrust::system::hip is the namespace containing functionality for allocating, manipulating,
 *         and deallocating memory available to Thrust's hip backend system.
 *         The identifiers are provided in a separate namespace underneath <tt>thrust::system</tt>
 *         for import convenience but are also aliased in the top-level <tt>thrust::hip</tt>
 *         namespace for easy access.
 *
 */


/*! \p pointer stores a pointer to an object allocated in memory available to the hip system.
 *  This type provides type safety when dispatching standard algorithms on ranges resident
 *  in hip memory.
 *
 *  \p pointer has pointer semantics: it may be dereferenced and manipulated with pointer arithmetic.
 *
 *  \p pointer can be created with the function \p hip::malloc, or by explicitly calling its constructor
 *  with a raw pointer.
 *
 *  The raw pointer encapsulated by a \p pointer may be obtained by eiter its <tt>get</tt> member function
 *  or the \p raw_pointer_cast function.
 *
 *  \note \p pointer is not a "smart" pointer; it is the programmer's responsibility to deallocate memory
 *  pointed to by \p pointer.
 *
 *  \tparam T specifies the type of the pointee.
 *
 *  \see hip::malloc
 *  \see hip::free
 *  \see raw_pointer_cast
 */

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
 *          allocated memory. A null <tt>hip::pointer<T></tt> is returned if
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

// XXX upon c++11
// template<typename T>
// using allocator = thrust::mr::stateless_resource_allocator<T,memory_resource >;
//

/*! \p hip::allocator is the default allocator used by the \p hip system's containers such as
 *  <tt>hip::vector</tt> if no user-specified allocator is provided. \p hip::allocator allocates
 *  (deallocates) storage with \p hip::malloc (\p hip::free).
 */
template <typename T>
struct allocator
    : thrust::mr::stateless_resource_allocator<
        T,
        system::hip::memory_resource
    >
{
private:
    typedef thrust::mr::stateless_resource_allocator<
        T,
        system::hip::memory_resource
    > base;

public:
    /*! The \p rebind metafunction provides the type of an \p allocator
     *  instantiated with another type.
     *
     *  \tparam U The other type to use for instantiation.
     */
    template <typename U>
    struct rebind
    {
        /*! The typedef \p other gives the type of the rebound \p allocator.
         */
        typedef allocator<U> other;
    };

    /*! No-argument constructor has no effect.
     */
    __host__ __device__
    inline allocator() {}

    /*! Copy constructor via base
     */
    __host__ __device__
    inline allocator(const allocator & other) : base(other) {}


    /*! Constructor from other \p allocator via base
     */
    template <typename U>
    __host__ __device__
    inline allocator(const allocator<U> & other) : base(other) {}

#if THRUST_CPP_DIALECT >= 2011
      allocator & operator=(const allocator &) = default;
#endif

    /*! Destructor has no effect.
     */
    __host__ __device__
    inline ~allocator() {}
}; // struct allocator

} // namespace hip_rocprim

namespace system
{

/*! \namespace thrust::hip
 *  \brief \p thrust::hip is a top-level alias for thrust::system::hip.
 */

namespace hip
{
    using thrust::hip_rocprim::allocator;
    using thrust::hip_rocprim::free;
    using thrust::hip_rocprim::malloc;
} // namespace hip
} /// namespace system

namespace hip
{
    using thrust::hip_rocprim::allocator;
    using thrust::hip_rocprim::free;
    using thrust::hip_rocprim::malloc;
} // end hip

THRUST_END_NS

#include <thrust/system/hip/detail/memory.inl>

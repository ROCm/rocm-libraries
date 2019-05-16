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

/*! \file thrust/system/hip/memory.h
 *  \brief Managing memory associated with Thrust's hip system.
 */

#pragma once

#include <hip/hip_runtime.h>

#include <ostream>
#include <thrust/detail/config.h>
#include <thrust/system/hip/execution_policy.h>
#include <thrust/memory.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/allocator/malloc_allocator.h>

BEGIN_NS_THRUST
namespace hip_rocprim
{

    template <typename>
    class pointer;

} // end hip_rocprim
END_NS_THRUST

// specialize thrust::iterator_traits to avoid problems with the name of
// pointer's constructor shadowing its nested pointer type
// do this before pointer is defined so the specialization is correctly
// used inside the definition
BEGIN_NS_THRUST

template <typename Element>
struct iterator_traits<thrust::hip_rocprim::pointer<Element>>
{
private:
    typedef thrust::hip_rocprim::pointer<Element> ptr;

public:
    typedef typename ptr::iterator_category iterator_category;
    typedef typename ptr::value_type        value_type;
    typedef typename ptr::difference_type   difference_type;
    typedef ptr                             pointer;
    typedef typename ptr::reference         reference;
}; // end iterator_traits

namespace hip_rocprim
{

// forward declaration of reference for pointer
template <typename Element>
class reference;

template <typename T>
class pointer : public thrust::pointer<T,
                                       thrust::hip_rocprim::tag,
                                       thrust::hip_rocprim::reference<T>,
                                       thrust::hip_rocprim::pointer<T>>
{

private:
    typedef thrust::pointer<T,
                            thrust::hip_rocprim::tag,
                            thrust::hip_rocprim::reference<T>,
                            thrust::hip_rocprim::pointer<T>>
        super_t;

public:
    __host__ __device__ pointer()
        : super_t()
    {
    }

    template <typename OtherT>
    __host__ __device__ explicit pointer(OtherT* ptr)
        : super_t(ptr)
    {
    }

    // STREAMHPC Fixes HCC linkage error
    __host__ __device__ explicit pointer(T* ptr)
        : super_t(ptr)
    {
    }

    template <typename OtherPointer>
    __host__ __device__
             pointer(const OtherPointer& other,
                     typename thrust::detail::enable_if_pointer_is_convertible<OtherPointer,
                                                                          pointer>::type* = 0)
        : super_t(other)
    {
    }

    template <typename OtherPointer>
    __host__ __device__ typename thrust::detail::
        enable_if_pointer_is_convertible<OtherPointer, pointer, pointer&>::type
        operator=(const OtherPointer& other)
    {
        return super_t::operator=(other);
    }
}; // struct pointer

template <typename T>
class reference : public thrust::reference<T,
                                           thrust::hip_rocprim::pointer<T>,
                                           thrust::hip_rocprim::reference<T>>
{

private:
    typedef thrust::
        reference<T, thrust::hip_rocprim::pointer<T>, thrust::hip_rocprim::reference<T>>
            super_t;

public:
    typedef typename super_t::value_type value_type;
    typedef typename super_t::pointer    pointer;

    __host__ __device__ explicit reference(const pointer& ptr)
        : super_t(ptr)
    {
    }

    template <typename OtherT>
    __host__ __device__ reference(
        const reference<OtherT>& other,
        typename thrust::detail::enable_if_convertible<typename reference<OtherT>::pointer,
                                                       pointer>::type* = 0)
        : super_t(other)
    {
    }
    template <typename OtherT>
    __host__ __device__ reference& operator=(const reference<OtherT>& other);

    __host__ __device__ reference& operator=(const value_type& x);
}; // struct reference

template <typename T>
__host__ __device__ void swap(reference<T> x, reference<T> y);

inline __host__ __device__ pointer<void> malloc(std::size_t n);

template <typename T>
inline __host__ __device__ pointer<T> malloc(std::size_t n);

inline __host__ __device__ void free(pointer<void> ptr);

// XXX upon c++11
// template<typename T> using allocator =
// thrust::detail::malloc_allocator<T,tag,pointer<T> >;
//
template <typename T>
struct allocator : thrust::detail::malloc_allocator<T, tag, pointer<T>>
{
    template <typename U>
    struct rebind
    {
        typedef allocator<U> other;
    };

    __host__ __device__ inline allocator() {}

    __host__ __device__ inline allocator(const allocator&)
        : thrust::detail::malloc_allocator<T, tag, thrust::hip_rocprim::pointer<T>>()
    {
    }

    template <typename U>
    __host__ __device__ inline allocator(const allocator<U>&)
    {
    }

    __host__ __device__ inline ~allocator() {}
}; // struct allocator

} // namespace hip_rocprim

namespace system
{
namespace hip
{
    using thrust::hip_rocprim::allocator;
    using thrust::hip_rocprim::free;
    using thrust::hip_rocprim::malloc;
    using thrust::hip_rocprim::pointer;
    using thrust::hip_rocprim::reference;
    using thrust::hip_rocprim::swap;
} // namespace hip
} /// namespace system

namespace hip
{
    using thrust::hip_rocprim::allocator;
    using thrust::hip_rocprim::free;
    using thrust::hip_rocprim::malloc;
    using thrust::hip_rocprim::pointer;
    using thrust::hip_rocprim::reference;
} // end hip

END_NS_THRUST

#include <thrust/system/hip/detail/memory.inl>

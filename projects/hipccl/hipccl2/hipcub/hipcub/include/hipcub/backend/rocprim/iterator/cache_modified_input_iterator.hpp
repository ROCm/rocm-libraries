/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2017-2020, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#ifndef HIPCUB_ROCPRIM_ITERATOR_CACHE_MODIFIED_INPUT_ITERATOR_HPP_
#define HIPCUB_ROCPRIM_ITERATOR_CACHE_MODIFIED_INPUT_ITERATOR_HPP_

#include <iterator>
#include <iostream>

#include "../thread/thread_load.hpp"
#include "../util_type.hpp"

#if (THRUST_VERSION >= 100700)
    // This iterator is compatible with Thrust API 1.7 and newer
    #include <thrust/iterator/iterator_facade.h>
    #include <thrust/iterator/iterator_traits.h>
#endif // THRUST_VERSION

BEGIN_HIPCUB_NAMESPACE

template <
    CacheLoadModifier   MODIFIER,
    typename            ValueType,
    typename            OffsetT = ptrdiff_t>
class CacheModifiedInputIterator
{
public:

    // Required iterator traits
    typedef CacheModifiedInputIterator          self_type;              ///< My own type
    typedef OffsetT                             difference_type;        ///< Type to express the result of subtracting one iterator from another
    typedef ValueType                           value_type;             ///< The type of the element the iterator can point to
    typedef ValueType*                          pointer;                ///< The type of a pointer to an element the iterator can point to
    typedef ValueType                           reference;              ///< The type of a reference to an element the iterator can point to
    typedef std::random_access_iterator_tag     iterator_category;      ///< The iterator category

public:

    /// Wrapped native pointer
    ValueType* ptr;

    /// Constructor
    __host__ __device__ __forceinline__ CacheModifiedInputIterator(
        ValueType* ptr)     ///< Native pointer to wrap
    :
        ptr(const_cast<typename std::remove_cv<ValueType>::type *>(ptr))
    {}

    /// Postfix increment
    __host__ __device__ __forceinline__ self_type operator++(int)
    {
        self_type retval = *this;
        ptr++;
        return retval;
    }

    /// Prefix increment
    __host__ __device__ __forceinline__ self_type operator++()
    {
        ptr++;
        return *this;
    }

    /// Indirection
    __device__ __forceinline__ reference operator*() const
    {
        return ThreadLoad<MODIFIER>(ptr);
    }

    /// Addition
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type operator+(Distance n) const
    {
        self_type retval(ptr + n);
        return retval;
    }

    /// Addition assignment
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator+=(Distance n)
    {
        ptr += n;
        return *this;
    }

    /// Subtraction
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type operator-(Distance n) const
    {
        self_type retval(ptr - n);
        return retval;
    }

    /// Subtraction assignment
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator-=(Distance n)
    {
        ptr -= n;
        return *this;
    }

    /// Distance
    __host__ __device__ __forceinline__ difference_type operator-(self_type other) const
    {
        return ptr - other.ptr;
    }

    /// Array subscript
    template <typename Distance>
    __device__ __forceinline__ reference operator[](Distance n) const
    {
        return ThreadLoad<MODIFIER>(ptr + n);
    }

    /// Structure dereference
    __device__ __forceinline__ pointer operator->()
    {
        return &ThreadLoad<MODIFIER>(ptr);
    }

    /// Equal to
    __host__ __device__ __forceinline__ bool operator==(const self_type& rhs)
    {
        return (ptr == rhs.ptr);
    }

    /// Not equal to
    __host__ __device__ __forceinline__ bool operator!=(const self_type& rhs)
    {
        return (ptr != rhs.ptr);
    }

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

    /// ostream operator
    friend std::ostream& operator<<(std::ostream& os, const self_type& /*itr*/)
    {
        return os;
    }

#endif

};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_ITERATOR_CACHE_MODIFIED_INPUT_ITERATOR_HPP_

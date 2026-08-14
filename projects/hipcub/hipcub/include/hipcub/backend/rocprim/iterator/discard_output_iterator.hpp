/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2020-2025, Advanced Micro Devices, Inc.  All rights reserved.
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

#ifndef HIPCUB_ROCPRIM_ITERATOR_DISCARD_OUTPUT_ITERATOR_HPP_
#define HIPCUB_ROCPRIM_ITERATOR_DISCARD_OUTPUT_ITERATOR_HPP_

#include "../../../config.hpp"
#include "../../../util_deprecated.hpp"

#include "iterator_category.hpp"

#include <rocprim/iterator/discard_iterator.hpp> // IWYU pragma: export

#include <iterator>
#include <iostream>

BEGIN_HIPCUB_NAMESPACE

/**
 * \addtogroup UtilIterator
 * @{
 */


/**
 * \brief A discard iterator
 */
template<typename OffsetT = ptrdiff_t>
class HIPCUB_DEPRECATED_BECAUSE(
    "Use rocprim::discard_iterator or rocthrust::discard_iterator instead") DiscardOutputIterator
{
public:
    // Required iterator traits
    using self_type = DiscardOutputIterator; ///< My own type
    using difference_type
        = OffsetT; ///< Type to express the result of subtracting one iterator from another
    using value_type = void; ///< The type of the element the iterator can point to
    using pointer    = void; ///< The type of a pointer to an element the iterator can point to
    using reference  = void; ///< The type of a reference to an element the iterator can point to
    using iterator_category =
        typename detail::IteratorCategory<value_type, reference>::type; ///< The iterator category

private:

    OffsetT offset;

public:

    /// Constructor
    __host__ __device__ __forceinline__ DiscardOutputIterator(
        OffsetT offset = 0)     ///< Base offset
    :
        offset(offset)
    {}

    /**
    * @typedef self_type
    * @brief Postfix increment
    */
    __host__ __device__ __forceinline__ self_type operator++(int)
    {
        self_type retval = *this;
        offset++;
        return retval;
    }

    /**
    * @typedef self_type
    * @brief Postfix increment
    */
    __host__ __device__ __forceinline__ self_type operator++()
    {
        offset++;
        return *this;
    }

    /**
    * @typedef self_type
    * @brief Indirection
    */
    __host__ __device__ __forceinline__ self_type& operator*()
    {
        // return self reference, which can be assigned to anything
        return *this;
    }

    /**
    * @typedef self_type
    * @brief Addition
    */
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type operator+(Distance n) const
    {
        self_type retval(offset + n);
        return retval;
    }

    /**
    * @typedef self_type
    * @brief Addition assignment
    */
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator+=(Distance n)
    {
        offset += n;
        return *this;
    }

    /**
    * @typedef self_type
    * @brief Subtraction assignment
    */
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type operator-(Distance n) const
    {
        self_type retval(offset - n);
        return retval;
    }

    /**
    * @typedef self_type
    * @brief Subtraction assignment
    */
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator-=(Distance n)
    {
        offset -= n;
        return *this;
    }

    /**
    * @typedef self_type
    * @brief Distance
    */
    __host__ __device__ __forceinline__ difference_type operator-(self_type other) const
    {
        return offset - other.offset;
    }

    /**
    * @typedef self_type
    * @brief Array subscript
    */
    template <typename Distance>
    __host__ __device__ __forceinline__ self_type& operator[](Distance)
    {
        // return self reference, which can be assigned to anything
        return *this;
    }

    /// Structure dereference
    __host__ __device__ __forceinline__ pointer operator->()
    {
        return;
    }

    /// Assignment to anything else (no-op)
    template<typename T>
    __host__ __device__ __forceinline__ void operator=(T const&)
    {}

    /// Cast to void* operator
    __host__ __device__ __forceinline__ operator void*() const
    {
        return nullptr;
    }

    /**
    * @typedef self_type
    * @brief Equal to
    */
    __host__ __device__ __forceinline__ bool operator==(const self_type& rhs) const
    {
        return (offset == rhs.offset);
    }

    /**
    * @typedef self_type
    * @brief Not equal to
    */
    __host__ __device__ __forceinline__ bool operator!=(const self_type& rhs) const
    {
        return (offset != rhs.offset);
    }

    /**
    * @typedef self_type
    * @brief ostream operator
    */
    HIPCUB_CLANG_SUPPRESS_DEPRECATED_PUSH
    friend std::ostream& operator<<(std::ostream& os, const self_type& itr)
    {
        os << "[" << itr.offset << "]";
        return os;
    }
    HIPCUB_CLANG_SUPPRESS_DEPRECATED_POP
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_ITERATOR_DISCARD_OUTPUT_ITERATOR_HPP_

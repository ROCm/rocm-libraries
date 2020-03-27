// Copyright (c) 2017-2019 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCPRIM_ITERATOR_ARG_INDEX_ITERATOR_HPP_
#define ROCPRIM_ITERATOR_ARG_INDEX_ITERATOR_HPP_

#include <iterator>
#include <iostream>
#include <cstddef>
#include <type_traits>

#include "../config.hpp"
#include "../types/key_value_pair.hpp"

/// \addtogroup iteratormodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \class arg_index_iterator
/// \brief A random-access input (read-only) iterator adaptor for pairing dereferenced values
/// with their indices.
///
/// \par Overview
/// * Dereferencing arg_index_iterator return a value of \p key_value_pair<Difference, InputValueType>
/// type, which includes value from the underlying range and its index in that range.
/// * \p std::iterator_traits<InputIterator>::value_type should be convertible to \p InputValueType.
///
/// \tparam InputIterator - type of the underlying random-access input iterator. Must be
/// a random-access iterator.
/// \tparam Difference - type used for identify distance between iterators and as the index type
/// in the output pair type (see \p value_type).
/// \tparam InputValueType - value type used in the output pair type (see \p value_type).
template<
    class InputIterator,
    class Difference = std::ptrdiff_t,
    class InputValueType = typename std::iterator_traits<InputIterator>::value_type
>
class arg_index_iterator
{
private:
    using input_category = typename std::iterator_traits<InputIterator>::iterator_category;

public:
    /// The type of the value that can be obtained by dereferencing the iterator.
    using value_type = ::rocprim::key_value_pair<Difference, InputValueType>;
    /// \brief A reference type of the type iterated over (\p value_type).
    /// It's `const` since arg_index_iterator is a read-only iterator.
    using reference = const value_type&;
    /// \brief A pointer type of the type iterated over (\p value_type).
    /// It's `const` since arg_index_iterator is a read-only iterator.
    using pointer = const value_type*;
    /// A type used for identify distance between iterators.
    using difference_type = Difference;
    /// The category of the iterator.
    using iterator_category = std::random_access_iterator_tag;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    using self_type = arg_index_iterator;
#endif

    static_assert(
        std::is_same<input_category, iterator_category>::value,
        "InputIterator must be a random-access iterator"
    );

    ROCPRIM_HOST_DEVICE inline
    ~arg_index_iterator() = default;

    /// \brief Creates a new arg_index_iterator.
    ///
    /// \param iterator input iterator pointing to the input range.
    /// \param offset index of the \p iterator in the input range.
    ROCPRIM_HOST_DEVICE inline
    arg_index_iterator(InputIterator iterator, difference_type offset = 0)
        : iterator_(iterator), offset_(offset)
    {
    }

    ROCPRIM_HOST_DEVICE inline
    arg_index_iterator& operator++()
    {
        iterator_++;
        offset_++;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    arg_index_iterator operator++(int)
    {
        arg_index_iterator old_ai = *this;
        iterator_++;
        offset_++;
        return old_ai;
    }

    ROCPRIM_HOST_DEVICE inline
    value_type operator*() const
    {
        value_type ret(offset_, *iterator_);
        return ret;
    }

    ROCPRIM_HOST_DEVICE inline
    pointer operator->() const
    {
        return &(*(*this));
    }

    ROCPRIM_HOST_DEVICE inline
    arg_index_iterator operator+(difference_type distance) const
    {
        return arg_index_iterator(iterator_ + distance, offset_ + distance);
    }

    ROCPRIM_HOST_DEVICE inline
    arg_index_iterator& operator+=(difference_type distance)
    {
        iterator_ += distance;
        offset_ += distance;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    arg_index_iterator operator-(difference_type distance) const
    {
        return arg_index_iterator(iterator_ - distance, offset_ - distance);
    }

    ROCPRIM_HOST_DEVICE inline
    arg_index_iterator& operator-=(difference_type distance)
    {
        iterator_ -= distance;
        offset_ -= distance;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    difference_type operator-(arg_index_iterator other) const
    {
        return iterator_ - other.iterator_;
    }

    ROCPRIM_HOST_DEVICE inline
    value_type operator[](difference_type distance) const
    {
        arg_index_iterator i = (*this) + distance;
        return *i;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator==(arg_index_iterator other) const
    {
        return (iterator_ == other.iterator_) && (offset_ == other.offset_);
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator!=(arg_index_iterator other) const
    {
        return (iterator_ != other.iterator_) || (offset_ != other.offset_);
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator<(arg_index_iterator other) const
    {
        return (iterator_ - other.iterator_) > 0;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator<=(arg_index_iterator other) const
    {
        return (iterator_ - other.iterator_) >= 0;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator>(arg_index_iterator other) const
    {
        return (iterator_ - other.iterator_) < 0;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator>=(arg_index_iterator other) const
    {
        return (iterator_ - other.iterator_) <= 0;
    }

    ROCPRIM_HOST_DEVICE inline
    void normalize()
    {
        offset_ = 0;
    }

    friend std::ostream& operator<<(std::ostream& os, const arg_index_iterator& /* iter */)
    {
        return os;
    }

private:
    InputIterator iterator_;
    difference_type offset_;
};

template<
    class InputIterator,
    class Difference,
    class InputValueType
>
ROCPRIM_HOST_DEVICE inline
arg_index_iterator<InputIterator, Difference, InputValueType>
operator+(typename arg_index_iterator<InputIterator, Difference, InputValueType>::difference_type distance,
          const arg_index_iterator<InputIterator, Difference, InputValueType>& iterator)
{
    return iterator + distance;
}


/// make_arg_index_iterator creates a arg_index_iterator using \p iterator as
/// the underlying iterator and \p offset as the position (index) of \p iterator
/// in the input range.
///
/// \tparam InputIterator - type of the underlying random-access input iterator. Must be
/// a random-access iterator.
/// \tparam Difference - type used for identify distance between iterators and as the index type
/// in the output pair type (see \p value_type in arg_index_iterator).
/// \tparam InputValueType - value type used in the output pair type (see \p value_type
/// in arg_index_iterator).
///
/// \param iterator input iterator pointing to the input range.
/// \param offset index of the \p iterator in the input range.
template<
    class InputIterator,
    class Difference = std::ptrdiff_t,
    class InputValueType = typename std::iterator_traits<InputIterator>::value_type
>
ROCPRIM_HOST_DEVICE inline
arg_index_iterator<InputIterator, Difference, InputValueType>
make_arg_index_iterator(InputIterator iterator, Difference offset = 0)
{
    return arg_index_iterator<InputIterator, Difference, InputValueType>(iterator, offset);
}

END_ROCPRIM_NAMESPACE

/// @}
// end of group iteratormodule

#endif // ROCPRIM_ITERATOR_ARG_INDEX_ITERATOR_HPP_

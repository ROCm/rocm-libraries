// Copyright (c) 2017-2018 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_ITERATOR_CONSTANT_ITERATOR_HPP_
#define ROCPRIM_ITERATOR_CONSTANT_ITERATOR_HPP_

#include <iterator>
#include <iostream>
#include <cstddef>
#include <type_traits>

#include "../config.hpp"

/// \addtogroup iteratormodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \class constant_iterator
/// \brief A random-access input (read-only) iterator which generates a sequence
/// of homogeneous values.
///
/// \par Overview
/// * A constant_iterator represents a pointer into a range of same values.
/// * Using it for simulating a range filled with a sequence of same values saves
/// memory capacity and bandwidth.
///
/// \tparam ValueType - type of value that can be obtained by dereferencing the iterator.
/// \tparam Difference - a type used for identify distance between iterators
template<
    class ValueType,
    class Difference = std::ptrdiff_t
>
class constant_iterator
{
public:
    /// The type of the value that can be obtained by dereferencing the iterator.
    using value_type = typename std::remove_const<ValueType>::type;
    /// \brief A reference type of the type iterated over (\p value_type).
    /// It's same as `value_type` since constant_iterator is a read-only
    /// iterator and does not have underlying buffer.
    using reference = value_type; // constant_iterator is not writable
    /// \brief A pointer type of the type iterated over (\p value_type).
    /// It's `const` since constant_iterator is a read-only iterator.
    using pointer = const value_type*; // constant_iterator is not writable
    /// A type used for identify distance between iterators.
    using difference_type = Difference;
    /// The category of the iterator.
    using iterator_category = std::random_access_iterator_tag;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    using self_type = constant_iterator;
#endif

    /// \brief Creates constant_iterator and sets its initial value to \p value.
    ///
    /// \param value initial value
    /// \param index optional index for constant_iterator
    ROCPRIM_HOST_DEVICE inline
    explicit constant_iterator(const value_type value, const size_t index = 0)
        : value_(value), index_(index)
    {
    }

    ROCPRIM_HOST_DEVICE inline
    ~constant_iterator() = default;

    ROCPRIM_HOST_DEVICE inline
    value_type operator*() const
    {
        return value_;
    }

    ROCPRIM_HOST_DEVICE inline
    pointer operator->() const
    {
        return &value_;
    }

    ROCPRIM_HOST_DEVICE inline
    constant_iterator& operator++()
    {
        index_++;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    constant_iterator operator++(int)
    {
        constant_iterator old_ci = *this;
        index_++;
        return old_ci;
    }

    ROCPRIM_HOST_DEVICE inline
    constant_iterator& operator--()
    {
        index_--;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    constant_iterator operator--(int)
    {
        constant_iterator old_ci = *this;
        index_--;
        return old_ci;
    }

    ROCPRIM_HOST_DEVICE inline
    constant_iterator operator+(difference_type distance) const
    {
        return constant_iterator(value_, index_ + distance);
    }

    ROCPRIM_HOST_DEVICE inline
    constant_iterator& operator+=(difference_type distance)
    {
        index_ += distance;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    constant_iterator operator-(difference_type distance) const
    {
        return constant_iterator(value_, index_ - distance);
    }

    ROCPRIM_HOST_DEVICE inline
    constant_iterator& operator-=(difference_type distance)
    {
        index_ -= distance;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    difference_type operator-(constant_iterator other) const
    {
        return static_cast<difference_type>(index_ - other.index_);
    }

    // constant_iterator is not writable, so we don't return reference,
    // just something convertible to reference. That matches requirement
    // of RandomAccessIterator concept
    ROCPRIM_HOST_DEVICE inline
    value_type operator[](difference_type) const
    {
        return value_;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator==(constant_iterator other) const
    {
        return value_ == other.value_ && index_ == other.index_;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator!=(constant_iterator other) const
    {
        return !(*this == other);
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator<(constant_iterator other) const
    {
        return distance_to(other) > 0;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator<=(constant_iterator other) const
    {
        return distance_to(other) >= 0;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator>(constant_iterator other) const
    {
        return distance_to(other) < 0;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator>=(constant_iterator other) const
    {
        return distance_to(other) <= 0;
    }

    friend std::ostream& operator<<(std::ostream& os, const constant_iterator& iter)
    {
        os << "[" << iter.value_ << "]";
        return os;
    }

private:
    inline
    difference_type distance_to(const constant_iterator& other) const
    {
        return difference_type(other.index_) - difference_type(index_);
    }

    value_type value_;
    size_t index_;
};

template<
    class ValueType,
    class Difference
>
ROCPRIM_HOST_DEVICE inline
constant_iterator<ValueType, Difference>
operator+(typename constant_iterator<ValueType, Difference>::difference_type distance,
          const constant_iterator<ValueType, Difference>& iter)
{
    return iter + distance;
}

/// make_constant_iterator creates a constant_iterator with its initial value
/// set to \p value.
///
/// \tparam ValueType - type of value that can be obtained by dereferencing created iterator.
/// \tparam Difference - a type used for identify distance between constant_iterator iterators.
///
/// \param value - initial value for constant_iterator.
/// \param index - optional index for constant_iterator.
template<
    class ValueType,
    class Difference = std::ptrdiff_t
>
ROCPRIM_HOST_DEVICE inline
constant_iterator<ValueType, Difference>
make_constant_iterator(ValueType value, size_t index = 0)
{
    return constant_iterator<ValueType, Difference>(value, index);
}

END_ROCPRIM_NAMESPACE

/// @}
// end of group iteratormodule

#endif // ROCPRIM_ITERATOR_CONSTANT_ITERATOR_HPP_

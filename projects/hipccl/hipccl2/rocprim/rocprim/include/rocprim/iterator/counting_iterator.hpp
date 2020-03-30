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

#ifndef ROCPRIM_ITERATOR_COUNTING_ITERATOR_HPP_
#define ROCPRIM_ITERATOR_COUNTING_ITERATOR_HPP_

#include <iterator>
#include <iostream>
#include <cstddef>
#include <type_traits>

#include "../config.hpp"
#include "../type_traits.hpp"

/// \addtogroup iteratormodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \class counting_iterator
/// \brief A random-access input (read-only) iterator over a sequence of consecutive integer values.
///
/// \par Overview
/// * A counting_iterator represents a pointer into a range of sequentially increasing values.
/// * Using it for simulating a range filled with a sequence of consecutive values saves
/// memory capacity and bandwidth.
///
/// \tparam Incrementable - type of value that can be obtained by dereferencing the iterator.
/// \tparam Difference - a type used for identify distance between iterators
template<
    class Incrementable,
    class Difference = std::ptrdiff_t
>
class counting_iterator
{
public:
    /// The type of the value that can be obtained by dereferencing the iterator.
    using value_type = typename std::remove_const<Incrementable>::type;
    /// \brief A reference type of the type iterated over (\p value_type).
    /// It's same as `value_type` since constant_iterator is a read-only
    /// iterator and does not have underlying buffer.
    using reference = value_type; // counting_iterator is not writable
    /// \brief A pointer type of the type iterated over (\p value_type).
    /// It's `const` since counting_iterator is a read-only iterator.
    using pointer = const value_type*; // counting_iterator is not writable
    /// A type used for identify distance between iterators.
    using difference_type = Difference;
    /// The category of the iterator.
    using iterator_category = std::random_access_iterator_tag;

    static_assert(std::is_integral<value_type>::value, "Incrementable must be integral type");

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    using self_type = counting_iterator;
#endif

    ROCPRIM_HOST_DEVICE inline
    counting_iterator() = default;

    /// \brief Creates counting_iterator with its initial value initialized
    /// to its default value (usually 0).
    ROCPRIM_HOST_DEVICE inline
    ~counting_iterator() = default;

    /// \brief Creates counting_iterator and sets its initial value to \p value_.
    ///
    /// \param value initial value
    ROCPRIM_HOST_DEVICE inline
    explicit counting_iterator(const value_type value) : value_(value)
    {
    }

    ROCPRIM_HOST_DEVICE inline
    counting_iterator& operator++()
    {
        value_++;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    counting_iterator operator++(int)
    {
        counting_iterator old_ci = *this;
        value_++;
        return old_ci;
    }

    ROCPRIM_HOST_DEVICE inline
    counting_iterator& operator--()
    {
        value_--;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    counting_iterator operator--(int)
    {
        counting_iterator old_ci = *this;
        value_--;
        return old_ci;
    }

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
    counting_iterator operator+(difference_type distance) const
    {
        return counting_iterator(value_ + static_cast<value_type>(distance));
    }

    ROCPRIM_HOST_DEVICE inline
    counting_iterator& operator+=(difference_type distance)
    {
        value_ += static_cast<value_type>(distance);
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    counting_iterator operator-(difference_type distance) const
    {
        return counting_iterator(value_ - static_cast<value_type>(distance));
    }

    ROCPRIM_HOST_DEVICE inline
    counting_iterator& operator-=(difference_type distance)
    {
        value_ -= static_cast<value_type>(distance);
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    difference_type operator-(counting_iterator other) const
    {
        return static_cast<difference_type>(value_ - other.value_);
    }

    // counting_iterator is not writable, so we don't return reference,
    // just something convertible to reference. That matches requirement
    // of RandomAccessIterator concept
    ROCPRIM_HOST_DEVICE inline
    value_type operator[](difference_type distance) const
    {
        return value_ + static_cast<value_type>(distance);
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator==(counting_iterator other) const
    {
        return this->equal_value(value_, other.value_);
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator!=(counting_iterator other) const
    {
        return !(*this == other);
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator<(counting_iterator other) const
    {
        return distance_to(other) > 0;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator<=(counting_iterator other) const
    {
        return distance_to(other) >= 0;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator>(counting_iterator other) const
    {
        return distance_to(other) < 0;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator>=(counting_iterator other) const
    {
        return distance_to(other) <= 0;
    }

    friend std::ostream& operator<<(std::ostream& os, const counting_iterator& iter)
    {
        os << "[" << iter.value_ << "]";
        return os;
    }

private:
    template<class T>
    inline
    bool equal_value(const T& x, const T& y) const
    {
        return (x == y);
    }

    inline
    difference_type distance_to(const counting_iterator& other) const
    {
        return difference_type(other.value_) - difference_type(value_);
    }

    value_type value_;
};

template<
    class Incrementable,
    class Difference
>
ROCPRIM_HOST_DEVICE inline
counting_iterator<Incrementable, Difference>
operator+(typename counting_iterator<Incrementable, Difference>::difference_type distance,
          const counting_iterator<Incrementable, Difference>& iter)
{
    return iter + distance;
}

/// make_counting_iterator creates a counting_iterator with its initial value
/// set to \p value.
///
/// \tparam Incrementable - type of value that can be obtained by dereferencing created iterator.
/// \tparam Difference - a type used for identify distance between counting_iterator iterators.
///
/// \param value - initial value for counting_iterator.
template<
    class Incrementable,
    class Difference = std::ptrdiff_t
>
ROCPRIM_HOST_DEVICE inline
counting_iterator<Incrementable, Difference>
make_counting_iterator(Incrementable value)
{
    return counting_iterator<Incrementable, Difference>(value);
}

END_ROCPRIM_NAMESPACE

/// @}
// end of group iteratormodule

#endif // ROCPRIM_ITERATOR_COUNTING_ITERATOR_HPP_

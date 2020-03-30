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

#ifndef ROCPRIM_ITERATOR_TRANSFORM_ITERATOR_HPP_
#define ROCPRIM_ITERATOR_TRANSFORM_ITERATOR_HPP_

#include <iterator>
#include <cstddef>
#include <type_traits>

#include "../config.hpp"
#include "../detail/match_result_type.hpp"

/// \addtogroup iteratormodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \class transform_iterator
/// \brief A random-access input (read-only) iterator adaptor for transforming dereferenced values.
///
/// \par Overview
/// * A transform_iterator uses functor of type UnaryFunction to transform value obtained
/// by dereferencing underlying iterator.
/// * Using it for simulating a range filled with results of applying functor of type
/// \p UnaryFunction to another range saves memory capacity and/or bandwidth.
///
/// \tparam InputIterator - type of the underlying random-access input iterator. Must be
/// a random-access iterator.
/// \tparam UnaryFunction - type of the transform functor.
/// \tparam ValueType - type of value that can be obtained by dereferencing the iterator.
/// By default it is the return type of \p UnaryFunction.
template<
    class InputIterator,
    class UnaryFunction,
    class ValueType =
        typename ::rocprim::detail::invoke_result<
            UnaryFunction, typename std::iterator_traits<InputIterator>::value_type
        >::type
>
class transform_iterator
{
public:
    /// The type of the value that can be obtained by dereferencing the iterator.
    using value_type = ValueType;
    /// \brief A reference type of the type iterated over (\p value_type).
    /// It's `const` since transform_iterator is a read-only iterator.
    using reference = const value_type&;
    /// \brief A pointer type of the type iterated over (\p value_type).
    /// It's `const` since transform_iterator is a read-only iterator.
    using pointer = const value_type*;
    /// A type used for identify distance between iterators.
    using difference_type = typename std::iterator_traits<InputIterator>::difference_type;
    /// The category of the iterator.
    using iterator_category = std::random_access_iterator_tag;
    /// The type of unary function used to transform input range.
    using unary_function = UnaryFunction;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    using self_type = transform_iterator;
#endif

    ROCPRIM_HOST_DEVICE inline
    ~transform_iterator() = default;

    /// \brief Creates a new transform_iterator.
    ///
    /// \param iterator input iterator to iterate over and transform.
    /// \param transform unary function used to transform values obtained
    /// from range pointed by \p iterator.
    ROCPRIM_HOST_DEVICE inline
    transform_iterator(InputIterator iterator, UnaryFunction transform)
        : iterator_(iterator), transform_(transform)
    {
    }

    ROCPRIM_HOST_DEVICE inline
    transform_iterator& operator++()
    {
        iterator_++;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    transform_iterator operator++(int)
    {
        transform_iterator old = *this;
        iterator_++;
        return old;
    }

    ROCPRIM_HOST_DEVICE inline
    transform_iterator& operator--()
    {
        iterator_--;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    transform_iterator operator--(int)
    {
        transform_iterator old = *this;
        iterator_--;
        return old;
    }

    ROCPRIM_HOST_DEVICE inline
    value_type operator*() const
    {
        return transform_(*iterator_);
    }

    ROCPRIM_HOST_DEVICE inline
    pointer operator->() const
    {
        return &(*(*this));
    }

    ROCPRIM_HOST_DEVICE inline
    value_type operator[](difference_type distance) const
    {
        transform_iterator i = (*this) + distance;
        return *i;
    }

    ROCPRIM_HOST_DEVICE inline
    transform_iterator operator+(difference_type distance) const
    {
        return transform_iterator(iterator_ + distance, transform_);
    }

    ROCPRIM_HOST_DEVICE inline
    transform_iterator& operator+=(difference_type distance)
    {
        iterator_ += distance;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    transform_iterator operator-(difference_type distance) const
    {
        return transform_iterator(iterator_ - distance, transform_);
    }

    ROCPRIM_HOST_DEVICE inline
    transform_iterator& operator-=(difference_type distance)
    {
        iterator_ -= distance;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    difference_type operator-(transform_iterator other) const
    {
        return iterator_ - other.iterator_;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator==(transform_iterator other) const
    {
        return iterator_ == other.iterator_;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator!=(transform_iterator other) const
    {
        return iterator_ != other.iterator_;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator<(transform_iterator other) const
    {
        return iterator_ < other.iterator_;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator<=(transform_iterator other) const
    {
        return iterator_ <= other.iterator_;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator>(transform_iterator other) const
    {
        return iterator_ > other.iterator_;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator>=(transform_iterator other) const
    {
        return iterator_ >= other.iterator_;
    }

    friend std::ostream& operator<<(std::ostream& os, const transform_iterator& /* iter */)
    {
        return os;
    }

private:
    InputIterator iterator_;
    UnaryFunction transform_;
};

template<
    class InputIterator,
    class UnaryFunction,
    class ValueType
>
ROCPRIM_HOST_DEVICE inline
transform_iterator<InputIterator, UnaryFunction, ValueType>
operator+(typename transform_iterator<InputIterator, UnaryFunction, ValueType>::difference_type distance,
          const transform_iterator<InputIterator, UnaryFunction, ValueType>& iterator)
{
    return iterator + distance;
}

/// make_transform_iterator creates a transform_iterator using \p iterator as
/// the underlying iterator and \p transform as the unary function.
///
/// \tparam InputIterator - type of the underlying random-access input iterator.
/// \tparam UnaryFunction - type of the transform functor.
///
/// \param iterator - input iterator.
/// \param transform - transform functor to use in created transform_iterator.
/// \return A new transform_iterator object which transforms the range pointed
/// by \p iterator using \p transform functor.
template<
    class InputIterator,
    class UnaryFunction
>
ROCPRIM_HOST_DEVICE inline
transform_iterator<InputIterator, UnaryFunction>
make_transform_iterator(InputIterator iterator, UnaryFunction transform)
{
    return transform_iterator<InputIterator, UnaryFunction>(iterator, transform);
}

/// @}
// end of group iteratormodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_ITERATOR_TRANSFORM_ITERATOR_HPP_

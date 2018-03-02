// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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

BEGIN_ROCPRIM_NAMESPACE

template<
    class InputIterator,
    class UnaryFunction,
#if defined(__cpp_lib_is_invocable) && !defined(DOXYGEN_SHOULD_SKIP_THIS) // C++17
    class ValueType =
        typename std::invoke_result<
            UnaryFunction, typename std::iterator_traits<InputIterator>::value_type
        >::type
#else
    class ValueType =
        typename std::result_of<
            UnaryFunction(typename std::iterator_traits<InputIterator>::value_type)
        >::type
#endif
>
class transform_iterator
{
private:
    using input_category = typename std::iterator_traits<InputIterator>::iterator_category;
    static_assert(
        std::is_same<input_category, std::random_access_iterator_tag>::value,
        "InputIterator must be a random-access iterator"
    );

public:
    using value_type = ValueType;
    using reference = const value_type&;
    using pointer = const value_type*;
    using difference_type = typename std::iterator_traits<InputIterator>::difference_type;
    using iterator_category = std::random_access_iterator_tag;
    using unary_function = UnaryFunction;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    using self_type = transform_iterator;
#endif

    ROCPRIM_HOST_DEVICE inline
    ~transform_iterator() = default;

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

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_ITERATOR_TRANSFORM_ITERATOR_HPP_

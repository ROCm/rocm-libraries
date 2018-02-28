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

#ifndef ROCPRIM_ITERATOR_ARG_INDEX_ITERATOR_HPP_
#define ROCPRIM_ITERATOR_ARG_INDEX_ITERATOR_HPP_

#include <iterator>
#include <iostream>
#include <cstddef>
#include <type_traits>

#include "../config.hpp"
#include "../types/key_value_pair.hpp"

BEGIN_ROCPRIM_NAMESPACE

template<
    class InputIterator,
    class Difference = std::ptrdiff_t,
    class Value = typename std::iterator_traits<InputIterator>::value_type
>
class arg_index_iterator
{
private:
    using input_category = typename std::iterator_traits<InputIterator>::iterator_category;
    
public:
    using value_type = ::rocprim::key_value_pair<Difference, Value>;
    using reference = const value_type&;
    using pointer = const value_type*;
    using difference_type = Difference;
    using iterator_category = std::random_access_iterator_tag;
    static_assert(std::is_same<input_category, iterator_category>::value,
                  "InputIterator must be a random-access iterator");

    ROCPRIM_HOST_DEVICE inline
    arg_index_iterator() = default;

    ROCPRIM_HOST_DEVICE inline
    ~arg_index_iterator() = default;

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

private:
    InputIterator iterator_;
    difference_type offset_;
};

template<
    class InputIterator,
    class Difference,
    class Value
>
ROCPRIM_HOST_DEVICE inline
arg_index_iterator<InputIterator, Difference, Value>
operator+(typename arg_index_iterator<InputIterator, Difference, Value>::difference_type distance,
          const arg_index_iterator<InputIterator, Difference, Value>& iterator)
{
    return iterator + distance;
}

template<
    class InputIterator,
    class Difference = std::ptrdiff_t,
    class Value = typename std::iterator_traits<InputIterator>::value_type
>
ROCPRIM_HOST_DEVICE inline
arg_index_iterator<InputIterator, Difference, Value>
make_arg_index_iterator(InputIterator iterator, Difference offset = 0)
{
    return arg_index_iterator<InputIterator, Difference, Value>(iterator, offset);
}

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_ITERATOR_ARG_INDEX_ITERATOR_HPP_

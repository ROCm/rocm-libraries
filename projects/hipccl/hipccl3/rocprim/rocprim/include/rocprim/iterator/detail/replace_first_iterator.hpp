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

#ifndef ROCPRIM_ITERATOR_REPLACE_FIRST_ITERATOR_HPP_
#define ROCPRIM_ITERATOR_REPLACE_FIRST_ITERATOR_HPP_

#include <iterator>
#include <cstddef>
#include <type_traits>

#include "../../config.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

// Replaces first value of given range with given value. Used in exclusive scan-by-key
// and exclusive segmented scan to avoid allocating additional memory and/or running
// additional kernels.
//
// Important: it does not dereference the first item in given range, so it does not matter
// if it's an invalid pointer.
//
// Usage:
// * input - start of your input range
// * value - value that should be used as first element of new range.
//
// replace_first_iterator<InputIterator>(input - 1, value);
//
// (input - 1) will never be dereferenced.
template<class InputIterator>
class replace_first_iterator
{
private:
    using input_category = typename std::iterator_traits<InputIterator>::iterator_category;
    static_assert(
        std::is_same<input_category, std::random_access_iterator_tag>::value,
        "InputIterator must be a random-access iterator"
    );

public:
    using value_type = typename std::iterator_traits<InputIterator>::value_type;
    using reference = value_type;
    using pointer = const value_type*;
    using difference_type = typename std::iterator_traits<InputIterator>::difference_type;
    using iterator_category = std::random_access_iterator_tag;

    ROCPRIM_HOST_DEVICE inline
    ~replace_first_iterator() = default;

    ROCPRIM_HOST_DEVICE inline
    replace_first_iterator(InputIterator iterator, value_type value, size_t index = 0)
        : iterator_(iterator), value_(value), index_(index)
    {
    }

    ROCPRIM_HOST_DEVICE inline
    replace_first_iterator& operator++()
    {
        iterator_++;
        index_++;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    replace_first_iterator operator++(int)
    {
        replace_first_iterator old = *this;
        iterator_++;
        index_++;
        return old;
    }

    ROCPRIM_HOST_DEVICE inline
    value_type operator*() const
    {
        if(index_ == 0)
        {
            return value_;
        }
        return *iterator_;
    }

    ROCPRIM_HOST_DEVICE inline
    value_type operator[](difference_type distance) const
    {
        replace_first_iterator i = (*this) + distance;
        return *i;
    }

    ROCPRIM_HOST_DEVICE inline
    replace_first_iterator operator+(difference_type distance) const
    {
        return replace_first_iterator(iterator_ + distance, value_, index_ + distance);
    }

    ROCPRIM_HOST_DEVICE inline
    replace_first_iterator& operator+=(difference_type distance)
    {
        iterator_ += distance;
        index_ += distance;
        return *this;
    }

private:
    InputIterator iterator_;
    value_type value_;
    size_t index_;
};

} // end of detail namespace

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_ITERATOR_REPLACE_FIRST_ITERATOR_HPP_

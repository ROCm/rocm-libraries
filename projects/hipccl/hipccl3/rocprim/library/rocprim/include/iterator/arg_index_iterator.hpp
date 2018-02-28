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

BEGIN_ROCPRIM_NAMESPACE

template<
    class IteratorT,
    class Difference = std::ptrdiff_t,
    class ValueT = typename std::iterator_traits<IteratorT>::value_type;
>
class arg_index_iterator
{
public:
    using difference_type = Difference;
    using value_type = ::rocprim::key_value_pair<Difference, ValueT>;
    using reference = value_type;
    using pointer = value_type*;
    using iterator_category = std::random_access_iterator_tag;
    
    ROCPRIM_HOST_DEVICE inline
    arg_index_iterator() = default;

    ROCPRIM_HOST_DEVICE inline
    ~arg_index_iterator() = default;
    
    ROCPRIM_HOST_DEVICE inline
    arg_index_iterator(IteratorT iter, difference_type offset = 0) : iter_(iter),
                                                                     offset_(offset)
    {
    }
    
    ROCPRIM_HOST_DEVICE inline
    arg_index_iterator& operator++()
    {
        offset_++;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    arg_index_iterator operator++(int)
    {
        arg_index_iterator old_ai = *this;
        offset_++;
        return old_ai;
    }

    ROCPRIM_HOST_DEVICE inline
    reference operator*() const
    {
        value_type ret(offset_, iter_[offset_]);
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
        return arg_index_iterator(iter_, offset_ + distance);
    }

    ROCPRIM_HOST_DEVICE inline
    arg_index_iterator& operator+=(difference_type distance)
    {
        offset_ += distance;
        return *this;
    }
    
    ROCPRIM_HOST_DEVICE inline
    arg_index_iterator operator-(difference_type distance) const
    {
        return arg_index_iterator(iter_, offset_ - distance);
    }

    ROCPRIM_HOST_DEVICE inline
    arg_index_iterator& operator-=(difference_type distance)
    {
        offset_ -= distance;
        return *this;
    }
    
    ROCPRIM_HOST_DEVICE inline
    difference_type operator-(arg_index_iterator other) const
    {
        return offset_ - other.offset_;
    }
    
    ROCPRIM_HOST_DEVICE inline
    reference operator[](difference_type distance) const
    {
        arg_index_iterator offset = (*this) + distance;
        return *offset;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator==(arg_index_iterator other) const
    {
        return (iter_ == other.iter_) && (offset_ == other.offset_);
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator!=(arg_index_iterator other) const
    {
        return (iter_ != other.iter_) || (offset_ != other.offset_);
    }
    
    ROCPRIM_HOST_DEVICE inline
    void normalize()
    {
        iter_ += offset_;
        offset_ = 0;
    }
    
    friend std::ostream& operator<<(std::ostream& os, const counting_iterator& iter)
    {
        os << "[" << iter.value_ << "]";
        return os;
    }
    
private:
    IteratorT iter_;
    difference_type offset_;
};

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_ITERATOR_ARG_INDEX_ITERATOR_HPP_

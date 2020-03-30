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

#ifndef ROCPRIM_ITERATOR_DISCARD_ITERATOR_HPP_
#define ROCPRIM_ITERATOR_DISCARD_ITERATOR_HPP_

#include <iterator>
#include <cstddef>
#include <type_traits>

#include "../config.hpp"

/// \addtogroup iteratormodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \class discard_iterator
/// \brief A random-access iterator which discards values assigned to it upon dereference.
///
/// \par Overview
/// * discard_iterator does not have any underlying array (memory) and does not save values
/// written to it upon dereference.
/// * discard_iterator can be used to safely ignore certain output of algorithms, which
/// saves memory capacity and/or bandwidth.
class discard_iterator
{
public:
    struct discard_value
    {
        ROCPRIM_HOST_DEVICE inline
        discard_value() = default;

        template<class T>
        ROCPRIM_HOST_DEVICE inline
        discard_value(T) {};

        ROCPRIM_HOST_DEVICE inline
        ~discard_value() = default;

        template<class T>
        ROCPRIM_HOST_DEVICE inline
        discard_value& operator=(const T&)
        {
            return *this;
        }
    };

    /// The type of the value that can be obtained by dereferencing the iterator.
    using value_type = discard_value;
    /// \brief A reference type of the type iterated over (\p value_type).
    using reference = discard_value;
    /// \brief A pointer type of the type iterated over (\p value_type).
    using pointer = discard_value*;
    /// A type used for identify distance between iterators.
    using difference_type = std::ptrdiff_t;
    /// The category of the iterator.
    using iterator_category = std::random_access_iterator_tag;

    /// \brief Creates a new discard_iterator.
    ///
    /// \param index - optional index of discard iterator (default = 0).
    ROCPRIM_HOST_DEVICE inline
    discard_iterator(size_t index = 0)
        : index_(index)
    {
    }

    ROCPRIM_HOST_DEVICE inline
    ~discard_iterator() = default;

    ROCPRIM_HOST_DEVICE inline
    discard_iterator& operator++()
    {
        index_++;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    discard_iterator operator++(int)
    {
        discard_iterator old = *this;
        index_++;
        return old;
    }

    ROCPRIM_HOST_DEVICE inline
    discard_iterator& operator--()
    {
        index_--;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    discard_iterator operator--(int)
    {
        discard_iterator old = *this;
        index_--;
        return old;
    }

    ROCPRIM_HOST_DEVICE inline
    discard_value operator*() const
    {
        return discard_value();
    }

    ROCPRIM_HOST_DEVICE inline
    discard_value operator[](difference_type distance) const
    {
        discard_iterator i = (*this) + distance;
        return *i;
    }

    ROCPRIM_HOST_DEVICE inline
    discard_iterator operator+(difference_type distance) const
    {
        auto i = static_cast<size_t>(static_cast<difference_type>(index_) + distance);
        return discard_iterator(i);
    }

    ROCPRIM_HOST_DEVICE inline
    discard_iterator& operator+=(difference_type distance)
    {
        index_ = static_cast<size_t>(static_cast<difference_type>(index_) + distance);
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    discard_iterator operator-(difference_type distance) const
    {
        auto i = static_cast<size_t>(static_cast<difference_type>(index_) - distance);
        return discard_iterator(i);
    }

    ROCPRIM_HOST_DEVICE inline
    discard_iterator& operator-=(difference_type distance)
    {
        index_ = static_cast<size_t>(static_cast<difference_type>(index_) - distance);
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    difference_type operator-(discard_iterator other) const
    {
        return index_ - other.index_;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator==(discard_iterator other) const
    {
        return index_ == other.index_;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator!=(discard_iterator other) const
    {
        return index_ != other.index_;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator<(discard_iterator other) const
    {
        return index_ < other.index_;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator<=(discard_iterator other) const
    {
        return index_ <= other.index_;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator>(discard_iterator other) const
    {
        return index_ > other.index_;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator>=(discard_iterator other) const
    {
        return index_ >= other.index_;
    }

    friend std::ostream& operator<<(std::ostream& os, const discard_iterator& /* iter */)
    {
        return os;
    }

private:
    mutable size_t index_;
};

ROCPRIM_HOST_DEVICE inline
discard_iterator
operator+(typename discard_iterator::difference_type distance,
          const discard_iterator& iterator)
{
    return iterator + distance;
}

/// make_discard_iterator creates a discard_iterator using optional
/// index parameter \p index.
///
/// \param index - optional index of discard iterator (default = 0).
/// \return A new discard_iterator object.
ROCPRIM_HOST_DEVICE inline
discard_iterator
make_discard_iterator(size_t index = 0)
{
    return discard_iterator(index);
}

/// @}
// end of group iteratormodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_ITERATOR_DISCARD_ITERATOR_HPP_

// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_ITERATOR_REVERSE_ITERATOR_HPP_
#define ROCPRIM_ITERATOR_REVERSE_ITERATOR_HPP_

#include <cstddef>
#include <iterator>
#include <type_traits>

#include "../config.hpp"

/// \addtogroup iteratormodule
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \class reverse_iterator
/// \brief A reverse iterator is an iterator adaptor that reverses the direction of a wrapped iterator.
///
/// \par Overview
/// * reverse_iterator can be used with random access iterators to reverse the direction of the iteration.
/// * The increment operators on the reverse iterator are mapped to decrements on the wrapped iterator,
/// * And the decrement operators on the reverse iterators are mapped to increments on the wrapped iterator.
/// * Use it to iterate over the elements of a container in reverse.
///
/// \tparam SourceIterator - type of the wrapped iterator.
template<class SourceIterator>
class reverse_iterator
{
public:
    static_assert(
        std::is_base_of<std::random_access_iterator_tag,
                        typename std::iterator_traits<SourceIterator>::iterator_category>::value,
        "SourceIterator must be a random access iterator");

    /// The type of the value that can be obtained by dereferencing the iterator.
    using value_type = typename std::iterator_traits<SourceIterator>::value_type;
    /// \brief A reference type of the type iterated over (\p value_type).
    using reference = typename std::iterator_traits<SourceIterator>::reference;
    /// \brief A pointer type of the type iterated over (\p value_type).
    using pointer = typename std::iterator_traits<SourceIterator>::pointer;
    /// A type used for identify distance between iterators.
    using difference_type = typename std::iterator_traits<SourceIterator>::difference_type;
    /// The category of the iterator.
    using iterator_category = std::random_access_iterator_tag;

    /// \brief Constructs a new reverse_iterator using the supplied source.
    ROCPRIM_HOST_DEVICE
    reverse_iterator(SourceIterator source_iterator) : source_iterator_(source_iterator) {}

    #ifndef DOXYGEN_SHOULD_SKIP_THIS
    ROCPRIM_HOST_DEVICE
    reverse_iterator& operator++()
    {
        --source_iterator_;
        return *this;
    }

    ROCPRIM_HOST_DEVICE
    reverse_iterator operator++(int)
    {
        reverse_iterator old = *this;
        --source_iterator_;
        return old;
    }

    ROCPRIM_HOST_DEVICE
    reverse_iterator& operator--()
    {
        ++source_iterator_;
        return *this;
    }

    ROCPRIM_HOST_DEVICE
    reverse_iterator operator--(int)
    {
        reverse_iterator old = *this;
        ++source_iterator_;
        return old;
    }

    ROCPRIM_HOST_DEVICE
    reference operator*()
    {
        return *(source_iterator_ - static_cast<difference_type>(1));
    }

    ROCPRIM_HOST_DEVICE
    reference operator[](difference_type distance)
    {
        reverse_iterator i = (*this) + distance;
        return *i;
    }

    ROCPRIM_HOST_DEVICE
    reverse_iterator operator+(difference_type distance) const
    {
        return reverse_iterator(source_iterator_ - distance);
    }

    ROCPRIM_HOST_DEVICE
    reverse_iterator& operator+=(difference_type distance)
    {
        source_iterator_ -= distance;
        return *this;
    }

    ROCPRIM_HOST_DEVICE
    reverse_iterator operator-(difference_type distance) const
    {
        return reverse_iterator(source_iterator_ + distance);
    }

    ROCPRIM_HOST_DEVICE
    reverse_iterator& operator-=(difference_type distance)
    {
        source_iterator_ += distance;
        return *this;
    }

    ROCPRIM_HOST_DEVICE
    difference_type operator-(reverse_iterator other) const
    {
        return other.source_iterator_ - source_iterator_;
    }

    ROCPRIM_HOST_DEVICE
    bool operator==(reverse_iterator other) const
    {
        return source_iterator_ == other.source_iterator_;
    }

    ROCPRIM_HOST_DEVICE
    bool operator!=(reverse_iterator other) const
    {
        return source_iterator_ != other.source_iterator_;
    }

    ROCPRIM_HOST_DEVICE
    bool operator<(reverse_iterator other) const
    {
        return other.source_iterator_ < source_iterator_;
    }

    ROCPRIM_HOST_DEVICE
    bool operator<=(reverse_iterator other) const
    {
        return other.source_iterator_ <= source_iterator_;
    }

    ROCPRIM_HOST_DEVICE
    bool operator>(reverse_iterator other) const
    {
        return other.source_iterator_ > source_iterator_;
    }

    ROCPRIM_HOST_DEVICE
    bool operator>=(reverse_iterator other) const
    {
        return other.source_iterator_ >= source_iterator_;
    }
    #endif // DOXYGEN_SHOULD_SKIP_THIS

private:
    SourceIterator source_iterator_;
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template<class SourceIterator>
ROCPRIM_HOST_DEVICE reverse_iterator<SourceIterator>
                    operator+(typename reverse_iterator<SourceIterator>::difference_type distance,
              const reverse_iterator<SourceIterator>&                    iterator)
{
    return iterator + distance;
}
#endif // DOXYGEN_SHOULD_SKIP_THIS

/// make_reverse_iterator creates a \p reverse_iterator wrapping \p source_iterator.
///
/// \tparam SourceIterator - type of \p source_iterator.
///
/// \param source_iterator - the iterator to wrap in the created \p reverse_iterator.
/// \return A \p reverse_iterator that wraps \p source_iterator.
template<class SourceIterator>
ROCPRIM_HOST_DEVICE reverse_iterator<SourceIterator>
                    make_reverse_iterator(SourceIterator source_iterator)
{
    return reverse_iterator<SourceIterator>(source_iterator);
}

END_ROCPRIM_NAMESPACE

/// @}
// end of group iteratormodule

#endif // ROCPRIM_ITERATOR_REVERSE_ITERATOR_HPP_

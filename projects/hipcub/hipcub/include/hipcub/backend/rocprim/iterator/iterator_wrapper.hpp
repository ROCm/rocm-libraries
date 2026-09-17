// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPCUB_ROCPRIM_WRAPPER_ITERATOR_HPP_
#define HIPCUB_ROCPRIM_WRAPPER_ITERATOR_HPP_

#include "../../../config.hpp"

#include <iterator>

BEGIN_HIPCUB_NAMESPACE

namespace detail
{

/// \class IteratorWrapper
/// \brief A wrapper for iterators to be able to make iterator_traits overwritable
///
/// \tparam WrappedIterator the iterator that is wrapped
/// \tparam DerivedIterator the iterator that this base class inherits
template<class WrappedIterator, class DerivedIterator>
class IteratorWrapper
{
public:
    using value_type        = typename WrappedIterator::value_type;
    using reference         = typename WrappedIterator::reference;
    using pointer           = typename WrappedIterator::pointer;
    using difference_type   = typename WrappedIterator::difference_type;
    using iterator_category = typename WrappedIterator::iterator_category;

    WrappedIterator iterator_;

    __host__ __device__ __forceinline__ IteratorWrapper(WrappedIterator iterator)
        : iterator_(iterator)
    {}

private:
    __host__ __device__ __forceinline__
    DerivedIterator& derived()
    {
        return static_cast<DerivedIterator&>(*this);
    }

public:
    __host__ __device__ __forceinline__
    DerivedIterator&
        operator++()
    {
        iterator_++;
        return derived();
    }

    __host__ __device__ __forceinline__
    DerivedIterator
        operator++(int)
    {
        DerivedIterator old_ci = derived();
        iterator_++;
        return old_ci;
    }

    __host__ __device__ __forceinline__
    DerivedIterator&
        operator--()
    {
        iterator_--;
        return derived();
    }

    __host__ __device__ __forceinline__
    DerivedIterator
        operator--(int)
    {
        DerivedIterator old_ci = derived();
        iterator_--;
        return old_ci;
    }

    __host__ __device__ __forceinline__
    value_type
        operator*() const
    {
        return iterator_.operator*();
    }

    __host__ __device__ __forceinline__
    pointer
        operator->() const
    {
        return iterator_.operator->();
    }

    __host__ __device__ __forceinline__
    value_type
        operator[](difference_type distance) const
    {
        return iterator_[distance];
    }

    __host__ __device__ __forceinline__
    DerivedIterator
        operator+(difference_type distance) const
    {
        return DerivedIterator(iterator_ + distance);
    }

    __host__ __device__ __forceinline__
    DerivedIterator&
        operator+=(difference_type distance)
    {
        iterator_ += distance;
        return derived();
    }

    __host__ __device__ __forceinline__
    DerivedIterator
        operator-(difference_type distance) const
    {
        return DerivedIterator(iterator_ - distance);
    }

    __host__ __device__ __forceinline__
    DerivedIterator&
        operator-=(difference_type distance)
    {
        iterator_ -= distance;
        return derived();
    }

    __host__ __device__ __forceinline__
    difference_type
        operator-(DerivedIterator other) const
    {
        return iterator_.operator-(other.iterator_);
    }

    __host__ __device__ __forceinline__
    bool operator==(DerivedIterator other) const
    {
        return iterator_ == other.iterator_;
    }

    __host__ __device__ __forceinline__
    bool operator!=(DerivedIterator other) const
    {
        return iterator_ != other.iterator_;
    }

    __host__ __device__ __forceinline__
    bool operator<(DerivedIterator other) const
    {
        return iterator_ < other.iterator_;
    }

    __host__ __device__ __forceinline__
    bool operator<=(DerivedIterator other) const
    {
        return iterator_ <= other.iterator_;
    }

    __host__ __device__ __forceinline__
    bool operator>(DerivedIterator other) const
    {
        return iterator_ > other.iterator_;
    }

    __host__ __device__ __forceinline__
    bool operator>=(DerivedIterator other) const
    {
        return iterator_ >= other.iterator_;
    }

    [[deprecated]]
    friend std::ostream&
        operator<<(std::ostream& os, const DerivedIterator& iter)
    {
        os << iter.iterator_;
        return os;
    }
};

} // namespace detail

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_WRAPPER_ITERATOR_HPP_

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

#ifndef TEST_IDENTITY_ITERATOR_HPP_
#define TEST_IDENTITY_ITERATOR_HPP_

#include <type_traits>
#include <iterator>

// rocPRIM
#include <rocprim/rocprim.hpp>

namespace test_utils
{

// Output iterator used in tests to check situtations when
// value_type of output iterator is void
template<class T>
class identity_iterator
{
public:
    // Iterator traits
    using difference_type = std::ptrdiff_t;
    using value_type = void;
    using pointer = void;
    using reference = T&;

    using iterator_category = std::random_access_iterator_tag;

    ROCPRIM_HOST_DEVICE inline
    identity_iterator(T * ptr)
        : ptr_(ptr)
    { }

    ROCPRIM_HOST_DEVICE inline
    ~identity_iterator() = default;

    ROCPRIM_HOST_DEVICE inline
    identity_iterator& operator++()
    {
        ptr_++;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    identity_iterator operator++(int)
    {
        identity_iterator old = *this;
        ptr_++;
        return old;
    }

    ROCPRIM_HOST_DEVICE inline
    identity_iterator& operator--()
    {
        ptr_--;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    identity_iterator operator--(int)
    {
        identity_iterator old = *this;
        ptr_--;
        return old;
    }

    ROCPRIM_HOST_DEVICE inline
    reference operator*() const
    {
        return *ptr_;
    }

    ROCPRIM_HOST_DEVICE inline
    reference operator[](difference_type n) const
    {
        return *(ptr_ + n);
    }

    ROCPRIM_HOST_DEVICE inline
    identity_iterator operator+(difference_type distance) const
    {
        auto i = ptr_ + distance;
        return identity_iterator(i);
    }

    ROCPRIM_HOST_DEVICE inline
    identity_iterator& operator+=(difference_type distance)
    {
        ptr_ += distance;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    identity_iterator operator-(difference_type distance) const
    {
        auto i = ptr_ - distance;
        return identity_iterator(i);
    }

    ROCPRIM_HOST_DEVICE inline
    identity_iterator& operator-=(difference_type distance)
    {
        ptr_ -= distance;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
    difference_type operator-(identity_iterator other) const
    {
        return ptr_ - other.ptr_;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator==(identity_iterator other) const
    {
        return ptr_ == other.ptr_;
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator!=(identity_iterator other) const
    {
        return ptr_ != other.ptr_;
    }

private:
    T* ptr_;
};

template<bool Wrap, class T>
inline
auto wrap_in_identity_iterator(T* ptr)
    -> typename std::enable_if<Wrap, identity_iterator<T>>::type
{
    return identity_iterator<T>(ptr);
}

template<bool Wrap, class T>
inline
auto wrap_in_identity_iterator(T* ptr)
    -> typename std::enable_if<!Wrap, T*>::type
{
    return ptr;
}

} // end test_utils namespace

#endif // TEST_IDENTITY_ITERATOR_HPP_

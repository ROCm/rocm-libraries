// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef TEST_INDIRECT_ITERATOR_HPP_
#define TEST_INDIRECT_ITERATOR_HPP_

#include <functional>
#include <type_traits>

#include <rocprim/config.hpp>

namespace test_utils
{

// assign-through reference_wrapper implementation
template<class T>
class reference_wrapper
{
public:
    // types
    using type = T;

    // construct/copy/destroy
    explicit constexpr reference_wrapper(T& t) : _ptr(&t) {}

    constexpr reference_wrapper(const reference_wrapper&) noexcept = default;

    // assignment
    constexpr reference_wrapper& operator=(const T& x) noexcept
    {
        *_ptr = x;
        return *this;
    }

    // access
    constexpr operator T&() const noexcept
    {
        return *_ptr;
    }
    constexpr T& get() const noexcept
    {
        return *_ptr;
    }

private:
    T* _ptr;
};

// Iterator used in tests to check situations when value_type of the
// iterator is not the same as the return type of operator[].
// It is a simplified version of device_vector::iterator from thrust.
template<class T>
class indirect_iterator
{
public:
    // Iterator traits
    using difference_type = std::ptrdiff_t;
    using value_type      = T;
    using pointer         = T*;
    using reference       = reference_wrapper<T>;

    using iterator_category = std::random_access_iterator_tag;

    ROCPRIM_HOST_DEVICE inline explicit indirect_iterator(T* ptr) : ptr_(ptr) {}

    ROCPRIM_HOST_DEVICE inline ~indirect_iterator() = default;

    ROCPRIM_HOST_DEVICE
    inline indirect_iterator&
        operator++()
    {
        ++ptr_;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline indirect_iterator operator++(int)
    {
        indirect_iterator old = *this;
        ++ptr_;
        return old;
    }

    ROCPRIM_HOST_DEVICE inline indirect_iterator& operator--()
    {
        --ptr_;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline indirect_iterator operator--(int)
    {
        indirect_iterator old = *this;
        --ptr_;
        return old;
    }

    ROCPRIM_HOST_DEVICE inline reference operator*() const
    {
        return *ptr_;
    }

    ROCPRIM_HOST_DEVICE inline reference operator[](difference_type n) const
    {
        return reference{*(ptr_ + n)};
    }

    ROCPRIM_HOST_DEVICE inline indirect_iterator operator+(difference_type distance) const
    {
        auto i = ptr_ + distance;
        return indirect_iterator{i};
    }

    ROCPRIM_HOST_DEVICE inline indirect_iterator& operator+=(difference_type distance)
    {
        ptr_ += distance;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline indirect_iterator operator-(difference_type distance) const
    {
        auto i = ptr_ - distance;
        return indirect_iterator{i};
    }

    ROCPRIM_HOST_DEVICE inline indirect_iterator& operator-=(difference_type distance)
    {
        ptr_ -= distance;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline difference_type operator-(indirect_iterator other) const
    {
        return ptr_ - other.ptr_;
    }

    ROCPRIM_HOST_DEVICE inline bool operator==(indirect_iterator other) const
    {
        return ptr_ == other.ptr_;
    }

    ROCPRIM_HOST_DEVICE inline bool operator!=(indirect_iterator other) const
    {
        return ptr_ != other.ptr_;
    }

private:
    T* ptr_;
};

template<bool Wrap, typename T>
inline auto wrap_in_indirect_iterator(T* ptr) ->
    typename std::enable_if<Wrap, indirect_iterator<T>>::type
{
    return indirect_iterator<T>(ptr);
}

template<bool Wrap, typename T>
inline auto wrap_in_indirect_iterator(T* ptr) -> typename std::enable_if<!Wrap, T*>::type
{
    return ptr;
}

} // namespace test_utils

#endif // TEST_INDIRECT_ITERATOR_HPP_

// Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_TEST_UTILS_CUSTOM_TEST_TYPES_HPP
#define ROCPRIM_TEST_UTILS_CUSTOM_TEST_TYPES_HPP

#include "../../common/utils_custom_type.hpp"

#include <rocprim/config.hpp>
#include <rocprim/functional.hpp>

#include <cstddef>
#include <ostream>
#include <type_traits>

namespace test_utils {

template<class T>
struct is_custom_test_array_type : std::false_type
{
};

template<class T>
struct inner_type
{
    using type = T;
};

template<class T>
struct custom_non_copyable_type
{
    T x;

    custom_non_copyable_type()                                           = default;
    custom_non_copyable_type(const custom_non_copyable_type&)            = delete;
    custom_non_copyable_type(custom_non_copyable_type&&)                 = default;
    ~custom_non_copyable_type()                                          = default;
    custom_non_copyable_type& operator=(const custom_non_copyable_type&) = delete;
    custom_non_copyable_type& operator=(custom_non_copyable_type&&)      = default;
};

template<class T>
struct custom_non_moveable_type
{
    T x;

    custom_non_moveable_type()                                           = default;
    custom_non_moveable_type(const custom_non_moveable_type&)            = delete;
    custom_non_moveable_type(custom_non_moveable_type&&)                 = delete;
    ~custom_non_moveable_type()                                          = default;
    custom_non_moveable_type& operator=(const custom_non_moveable_type&) = delete;
    custom_non_moveable_type& operator=(custom_non_moveable_type&&)      = delete;
};

template<class T>
struct custom_non_default_type
{
    T x;

    custom_non_default_type() = delete;
};

// Custom type used in tests
template<class T, size_t N>
struct custom_test_array_type
{
    using value_type = T;
    static constexpr size_t size = N;

    T values[N];

    ROCPRIM_HOST_DEVICE inline
        custom_test_array_type()
    {
        for(size_t i = 0; i < N; i++)
        {
            values[i] = T(i + 1);
        }
    }

    ROCPRIM_HOST_DEVICE inline custom_test_array_type(T v)
    {
        for(size_t i = 0; i < N; i++)
        {
            values[i] = v;
        }
    }

    template<class U>
    ROCPRIM_HOST_DEVICE inline
        custom_test_array_type(const custom_test_array_type<U, N>& other)
    {
        for(size_t i = 0; i < N; i++)
        {
            values[i] = other.values[i];
        }
    }

    ROCPRIM_HOST_DEVICE inline ~custom_test_array_type() {}

    ROCPRIM_HOST_DEVICE
    inline custom_test_array_type&
        operator=(const custom_test_array_type& other)
    {
        for(size_t i = 0; i < N; i++)
        {
            values[i] = other.values[i];
        }
        return *this;
    }

    ROCPRIM_HOST_DEVICE
    inline custom_test_array_type
        operator+(const custom_test_array_type& other) const
    {
        custom_test_array_type result;
        for(size_t i = 0; i < N; i++)
        {
            result.values[i] = values[i] + other.values[i];
        }
        return result;
    }

    ROCPRIM_HOST_DEVICE inline
        custom_test_array_type operator-(const custom_test_array_type& other) const
    {
        custom_test_array_type result;
        for(size_t i = 0; i < N; i++)
        {
            result.values[i] = values[i] - other.values[i];
        }
        return result;
    }

    ROCPRIM_HOST_DEVICE inline
        bool operator<(const custom_test_array_type& other) const
    {
        for(unsigned int i = 0; i < N; i++)
        {
            if(values[i] < other.values[i])
            {
                return true;
            }
            else if(other.values[i] < values[i])
            {
                return false;
            }
        }
        return false;
    }

    ROCPRIM_HOST_DEVICE
    inline bool
        operator>(const custom_test_array_type& other) const
    {
        for(unsigned int i = 0; i < N; i++)
        {
            if(values[i] > other.values[i])
            {
                return true;
            }
            else if(other.values[i] > values[i])
            {
                return false;
            }
        }
        return false;
    }

    ROCPRIM_HOST_DEVICE
    inline bool
        operator==(const custom_test_array_type& other) const
    {
        for(size_t i = 0; i < N; i++)
        {
            if(values[i] != other.values[i])
            {
                return false;
            }
        }
        return true;
    }

    ROCPRIM_HOST_DEVICE
    inline bool
        operator!=(const custom_test_array_type& other) const
    {
        return !(*this == other);
    }
};

template<class T, size_t N> inline
    std::ostream& operator<<(std::ostream& stream,
               const custom_test_array_type<T, N>& value)
{
    stream << "[";
    for(size_t i = 0; i < N; i++)
    {
        stream << value.values[i];
        if(i != N - 1)
        {
            stream << "; ";
        }
    }
    stream << "]";
    return stream;
}

template<class T, size_t N>
struct is_custom_test_array_type<custom_test_array_type<T, N>> : std::true_type
{
};

template<class T>
struct inner_type<common::custom_type<T, T, true>>
{
    using type = T;
};

template<class T, size_t N>
struct inner_type<custom_test_array_type<T, N>>
{
    using type = T;
};

template<class T>
struct inner_type<custom_non_copyable_type<T>>
{
    using type = T;
};

template<class T>
struct inner_type<custom_non_moveable_type<T>>
{
    using type = T;
};

template<class T>
struct inner_type<custom_non_default_type<T>>
{
    using type = T;
};
} // namespace test_utils

namespace common
{

template<class T>
struct is_custom_type<test_utils::custom_non_copyable_type<T>> : std::true_type
{};

template<class T>
struct is_custom_type<test_utils::custom_non_moveable_type<T>> : std::true_type
{};

template<class T>
struct is_custom_type<test_utils::custom_non_default_type<T>> : std::true_type
{};
} // namespace common

#endif //ROCPRIM_TEST_UTILS_CUSTOM_TEST_TYPES_HPP

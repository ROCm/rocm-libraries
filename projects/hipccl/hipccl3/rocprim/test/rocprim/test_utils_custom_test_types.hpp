// Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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

#include "test_utils_half.hpp"
#include "test_utils_bfloat16.hpp"

namespace test_utils {

template<class T>
struct is_custom_test_type : std::false_type
{
};

template<class T>
struct is_custom_test_array_type : std::false_type
{
};

template<class T>
struct inner_type
{
    using type = T;
};

// Custom type used in tests
template<class T>
struct custom_test_type
{
    using value_type = T;

    T x;
    T y;

    // Non-zero values in default constructor for checking reduce and scan:
    // ensure that scan_op(custom_test_type(), value) != value
    ROCPRIM_HOST_DEVICE inline
        custom_test_type() : x(12), y(34) {}

    ROCPRIM_HOST_DEVICE inline
        custom_test_type(T x, T y) : x(x), y(y) {}

    ROCPRIM_HOST_DEVICE inline
        custom_test_type(T xy) : x(xy), y(xy) {}

    template<class U>
    ROCPRIM_HOST_DEVICE inline
        custom_test_type(const custom_test_type<U>& other)
    {
        x = other.x;
        y = other.y;
    }

    ROCPRIM_HOST_DEVICE inline
        ~custom_test_type() {}

    ROCPRIM_HOST_DEVICE inline
        custom_test_type& operator=(const custom_test_type& other)
    {
        x = other.x;
        y = other.y;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
        custom_test_type operator+(const custom_test_type& other) const
    {
        return custom_test_type(x + other.x, y + other.y);
    }

    ROCPRIM_HOST_DEVICE inline
        custom_test_type operator-(const custom_test_type& other) const
    {
        return custom_test_type(x - other.x, y - other.y);
    }

    ROCPRIM_HOST_DEVICE inline
        bool operator<(const custom_test_type& other) const
    {
        return (x < other.x || (x == other.x && y < other.y));
    }

    ROCPRIM_HOST_DEVICE inline
        bool operator>(const custom_test_type& other) const
    {
        return (x > other.x || (x == other.x && y > other.y));
    }

    ROCPRIM_HOST_DEVICE inline
        bool operator==(const custom_test_type& other) const
    {
        return (x == other.x && y == other.y);
    }

    ROCPRIM_HOST_DEVICE inline
        bool operator!=(const custom_test_type& other) const
    {
        return !(*this == other);
    }
};

//Overload for rocprim::half
template<>
struct custom_test_type<rocprim::half>
{
    using value_type = rocprim::half;

    rocprim::half x;
    rocprim::half y;

    // Non-zero values in default constructor for checking reduce and scan:
    // ensure that scan_op(custom_test_type(), value) != value
    ROCPRIM_HOST_DEVICE inline
        custom_test_type() : x(12), y(34) {}

    ROCPRIM_HOST_DEVICE inline
        custom_test_type(rocprim::half x, rocprim::half y) : x(x), y(y) {}

    ROCPRIM_HOST_DEVICE inline
        custom_test_type(rocprim::half xy) : x(xy), y(xy) {}

    template<class U>
    ROCPRIM_HOST_DEVICE inline
        custom_test_type(const custom_test_type<U>& other)
    {
        x = static_cast<rocprim::half>(other.x);
        y = static_cast<rocprim::half>(other.y);
    }

    ROCPRIM_HOST_DEVICE inline
        ~custom_test_type() {}

    ROCPRIM_HOST_DEVICE inline
        custom_test_type& operator=(const custom_test_type& other)
    {
        x = other.x;
        y = other.y;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
        custom_test_type operator+(const custom_test_type& other) const
    {
        return custom_test_type(half_plus()(x, other.x), half_plus()(y, other.y));
    }

    ROCPRIM_HOST_DEVICE inline
        custom_test_type operator-(const custom_test_type& other) const
    {
        return custom_test_type(half_minus()(x, other.x), half_minus()(y, other.y));
    }

    ROCPRIM_HOST_DEVICE inline
        bool operator<(const custom_test_type& other) const
    {
        return (half_less()(x, other.x) || (half_equal_to()(x, other.x) && half_less()(y, other.y)));
    }

    ROCPRIM_HOST_DEVICE inline
        bool operator>(const custom_test_type& other) const
    {
        return (half_greater()(x, other.x) || (half_equal_to()(x, other.x) && half_greater()(y, other.y)));
    }

    ROCPRIM_HOST_DEVICE inline
        bool operator==(const custom_test_type& other) const
    {
        return (half_equal_to()(x, other.x) && half_equal_to()(y, other.y));
    }

    ROCPRIM_HOST_DEVICE inline
        bool operator!=(const custom_test_type& other) const
    {
        return !(*this == other);
    }
};

//Overload for rocprim::bfloat16
template<>
struct custom_test_type<rocprim::bfloat16>
{
    using value_type = rocprim::bfloat16;

    rocprim::bfloat16 x;
    rocprim::bfloat16 y;

    // Non-zero values in default constructor for checking reduce and scan:
    // ensure that scan_op(custom_test_type(), value) != value
    ROCPRIM_HOST_DEVICE inline
        custom_test_type() : x(12), y(34) {}

    ROCPRIM_HOST_DEVICE inline
        custom_test_type(rocprim::bfloat16 x, rocprim::bfloat16 y) : x(x), y(y) {}

    ROCPRIM_HOST_DEVICE inline
        custom_test_type(rocprim::bfloat16 xy) : x(xy), y(xy) {}

    template<class U>
    ROCPRIM_HOST_DEVICE inline
        custom_test_type(const custom_test_type<U>& other)
    {
        x = static_cast<rocprim::bfloat16>(other.x);
        y = static_cast<rocprim::bfloat16>(other.y);
    }

    ROCPRIM_HOST_DEVICE inline
        ~custom_test_type() {}

    ROCPRIM_HOST_DEVICE inline
        custom_test_type& operator=(const custom_test_type& other)
    {
        x = other.x;
        y = other.y;
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
        custom_test_type operator+(const custom_test_type& other) const
    {
        return custom_test_type(bfloat16_plus()(x, other.x), bfloat16_plus()(y, other.y));
    }

    ROCPRIM_HOST_DEVICE inline
        custom_test_type operator-(const custom_test_type& other) const
    {
        return custom_test_type(bfloat16_minus()(x, other.x), bfloat16_minus()(y, other.y));
    }

    ROCPRIM_HOST_DEVICE inline
        bool operator<(const custom_test_type& other) const
    {
        return (bfloat16_less()(x, other.x) || (bfloat16_equal_to()(x, other.x) && bfloat16_less()(y, other.y)));
    }

    ROCPRIM_HOST_DEVICE inline
        bool operator>(const custom_test_type& other) const
    {
        return (bfloat16_greater()(x, other.x) || (bfloat16_equal_to()(x, other.x) && bfloat16_greater()(y, other.y)));
    }

    ROCPRIM_HOST_DEVICE inline
        bool operator==(const custom_test_type& other) const
    {
        return (bfloat16_equal_to()(x, other.x) && bfloat16_equal_to()(y, other.y));
    }

    ROCPRIM_HOST_DEVICE inline
        bool operator!=(const custom_test_type& other) const
    {
        return !(*this == other);
    }
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

    ROCPRIM_HOST_DEVICE inline
        custom_test_array_type(T v)
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

    ROCPRIM_HOST_DEVICE inline
        ~custom_test_array_type() {}

    ROCPRIM_HOST_DEVICE inline
        custom_test_array_type& operator=(const custom_test_array_type& other)
    {
        for(size_t i = 0; i < N; i++)
        {
            values[i] = other.values[i];
        }
        return *this;
    }

    ROCPRIM_HOST_DEVICE inline
        custom_test_array_type operator+(const custom_test_array_type& other) const
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
        for(size_t i = 0; i < N; i++)
        {
            if(values[i] >= other.values[i])
            {
                return false;
            }
        }
        return true;
    }

    ROCPRIM_HOST_DEVICE inline
        bool operator>(const custom_test_array_type& other) const
    {
        for(size_t i = 0; i < N; i++)
        {
            if(values[i] <= other.values[i])
            {
                return false;
            }
        }
        return true;
    }

    ROCPRIM_HOST_DEVICE inline
        bool operator==(const custom_test_array_type& other) const
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

    ROCPRIM_HOST_DEVICE inline
        bool operator!=(const custom_test_array_type& other) const
    {
        return !(*this == other);
    }
};

template<class T> inline
    std::ostream& operator<<(std::ostream& stream,
               const custom_test_type<T>& value)
{
    stream << "[" << value.x << "; " << value.y << "]";
    return stream;
}

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

template<class T>
struct is_custom_test_type<custom_test_type<T>> : std::true_type
{
};

template<class T, size_t N>
struct is_custom_test_array_type<custom_test_array_type<T, N>> : std::true_type
{
};


template<class T>
struct inner_type<custom_test_type<T>>
{
    using type = T;
};

template<class T, size_t N>
struct inner_type<custom_test_array_type<T, N>>
{
    using type = T;
};
}
#endif //ROCPRIM_TEST_UTILS_CUSTOM_TEST_TYPES_HPP

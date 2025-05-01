// MIT License
//
// Copyright (c) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef HIPCUB_TEST_HIPCUB_TEST_UTILS_CUSTOM_TEST_TYPES_HPP_
#define HIPCUB_TEST_HIPCUB_TEST_UTILS_CUSTOM_TEST_TYPES_HPP_

#include "test_utils_bfloat16.hpp"
#include "test_utils_functional.hpp"
#include "test_utils_half.hpp"

namespace test_utils {

template<class T>
struct custom_test_type
{
    using value_type = T;

    T x;
    T y;

    HIPCUB_HOST_DEVICE inline
        constexpr custom_test_type() : x{}, y{} {}

    HIPCUB_HOST_DEVICE inline
        constexpr custom_test_type(T x, T y) : x(x), y(y) {}

    HIPCUB_HOST_DEVICE inline
        constexpr custom_test_type(T xy) : x(xy), y(xy) {}

    template<class U>
    HIPCUB_HOST_DEVICE inline
        custom_test_type(const custom_test_type<U>& other) : x(other.x), y(other.y)
    {
    }

#ifndef HIPCUB_CUB_API
    HIPCUB_HOST_DEVICE inline
        ~custom_test_type() = default;
#endif

    HIPCUB_HOST_DEVICE inline
        custom_test_type& operator=(const custom_test_type& other)
    {
        x = other.x;
        y = other.y;
        return *this;
    }

    HIPCUB_HOST_DEVICE inline
        custom_test_type operator+(const custom_test_type& other) const
    {
        return custom_test_type(x + other.x, y + other.y);
    }

    HIPCUB_HOST_DEVICE inline
        custom_test_type operator-(const custom_test_type& other) const
    {
        return custom_test_type(x - other.x, y - other.y);
    }

    HIPCUB_HOST_DEVICE inline
        bool operator<(const custom_test_type& other) const
    {
        return (x < other.x || (x == other.x && y < other.y));
    }

    HIPCUB_HOST_DEVICE inline
        bool operator>(const custom_test_type& other) const
    {
        return (x > other.x || (x == other.x && y > other.y));
    }

    HIPCUB_HOST_DEVICE inline
        bool operator==(const custom_test_type& other) const
    {
        return (x == other.x && y == other.y);
    }

    HIPCUB_HOST_DEVICE inline
        bool operator!=(const custom_test_type& other) const
    {
        return !(*this == other);
    }
};

template<class T>
inline std::ostream& operator<<(std::ostream& stream, const test_utils::custom_test_type<T>& value)
{
    stream << "[" << value.x << "; " << value.y << "]";
    return stream;
}

//Overload for test_utils::half
template<>
struct custom_test_type<test_utils::half>
{
    using value_type = test_utils::half;

    test_utils::half x;
    test_utils::half y;

    // Non-zero values in default constructor for checking reduce and scan:
    // ensure that scan_op(custom_test_type(), value) != value
    HIPCUB_HOST_DEVICE inline
        custom_test_type() : x(12), y(34) {}

    HIPCUB_HOST_DEVICE inline
        custom_test_type(test_utils::half x, test_utils::half y) : x(x), y(y) {}

    HIPCUB_HOST_DEVICE inline
        custom_test_type(test_utils::half xy) : x(xy), y(xy) {}

    template<class U>
    HIPCUB_HOST_DEVICE inline
        custom_test_type(const custom_test_type<U>& other) : x(other.x), y(other.y)
    {
    }

    HIPCUB_HOST_DEVICE inline
        ~custom_test_type() {}

    HIPCUB_HOST_DEVICE inline
        custom_test_type& operator=(const custom_test_type& other)
    {
        x = other.x;
        y = other.y;
        return *this;
    }

    HIPCUB_HOST_DEVICE inline
        custom_test_type operator+(const custom_test_type& other) const
    {
        return custom_test_type(half_plus()(x, other.x), half_plus()(y, other.y));
    }

    HIPCUB_HOST_DEVICE inline
        custom_test_type operator-(const custom_test_type& other) const
    {
        return custom_test_type(half_minus()(x, other.x), half_minus()(y, other.y));
    }

    HIPCUB_HOST_DEVICE inline
        bool operator<(const custom_test_type& other) const
    {
        return (test_utils::less()(x, other.x) || (half_equal_to()(x, other.x) && test_utils::less()(y, other.y)));
    }

    HIPCUB_HOST_DEVICE inline
        bool operator>(const custom_test_type& other) const
    {
        return (greater()(x, other.x) || (half_equal_to()(x, other.x) && greater()(y, other.y)));
    }

    HIPCUB_HOST_DEVICE inline
        bool operator==(const custom_test_type& other) const
    {
        return (half_equal_to()(x, other.x) && half_equal_to()(y, other.y));
    }

    HIPCUB_HOST_DEVICE inline
        bool operator!=(const custom_test_type& other) const
    {
        return !(*this == other);
    }
};

//Overload for test_utils::bfloat16
template<>
struct custom_test_type<test_utils::bfloat16>
{
    using value_type = test_utils::bfloat16;

    test_utils::bfloat16 x;
    test_utils::bfloat16 y;

    // Non-zero values in default constructor for checking reduce and scan:
    // ensure that scan_op(custom_test_type(), value) != value
    HIPCUB_HOST_DEVICE inline
        custom_test_type() : x(float(12)), y(float(34)) {}

    HIPCUB_HOST_DEVICE inline
        custom_test_type(test_utils::bfloat16 x, test_utils::bfloat16 y) : x(x), y(y) {}

    HIPCUB_HOST_DEVICE inline
        custom_test_type(test_utils::bfloat16 xy) : x(xy), y(xy) {}

    template<class U>
    HIPCUB_HOST_DEVICE inline
        custom_test_type(const custom_test_type<U>& other) : x(other.x), y(other.y)
    {
    }

    HIPCUB_HOST_DEVICE inline
        ~custom_test_type() {}

    HIPCUB_HOST_DEVICE inline
        custom_test_type& operator=(const custom_test_type& other)
    {
        x = other.x;
        y = other.y;
        return *this;
    }

    HIPCUB_HOST_DEVICE inline
        custom_test_type operator+(const custom_test_type& other) const
    {
        return custom_test_type(bfloat16_plus()(x, other.x), bfloat16_plus()(y, other.y));
    }

    HIPCUB_HOST_DEVICE inline
        custom_test_type operator-(const custom_test_type& other) const
    {
        return custom_test_type(bfloat16_minus()(x, other.x), bfloat16_minus()(y, other.y));
    }

    HIPCUB_HOST_DEVICE inline
        bool operator<(const custom_test_type& other) const
    {
        return (test_utils::less()(x, other.x) || (bfloat16_equal_to()(x, other.x) && test_utils::less()(y, other.y)));
    }

    HIPCUB_HOST_DEVICE inline
        bool operator>(const custom_test_type& other) const
    {
        return (greater()(x, other.x) || (bfloat16_equal_to()(x, other.x) && greater()(y, other.y)));
    }

    HIPCUB_HOST_DEVICE inline
        bool operator==(const custom_test_type& other) const
    {
        return (bfloat16_equal_to()(x, other.x) && bfloat16_equal_to()(y, other.y));
    }

    HIPCUB_HOST_DEVICE inline
        bool operator!=(const custom_test_type& other) const
    {
        return !(*this == other);
    }
};

template<class T>
struct is_custom_test_type : std::false_type
{
};

template<class T>
struct is_custom_test_type<custom_test_type<T>> : std::true_type
{
};

template <typename T>
struct inner_type {
    using type = T;
};

template <typename T>
struct inner_type<custom_test_type<T>> {
    using type = T;
};

} // end of test_utils namespace
#endif  // HIPCUB_TEST_HIPCUB_TEST_UTILS_CUSTOM_TEST_TYPES_HPP_

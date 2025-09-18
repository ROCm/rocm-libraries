// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef COMMON_UTILS_CUSTOM_TYPE_HPP_
#define COMMON_UTILS_CUSTOM_TYPE_HPP_

#include <rocprim/config.hpp>
#include <rocprim/functional.hpp>

#include <ostream>
#include <type_traits>

namespace common
{

template<typename T, typename U = T, bool NonZero = false>
struct custom_type
{
    using first_type  = T;
    using second_type = U;
    // value_type is valid if T == U
    using value_type = std::conditional_t<std::is_same<T, U>::value, T, void>;

    T x;
    U y;

    // Non-zero values in default constructor for checking reduce and scan:
    // ensure that scan_op(custom_type(), value) != value
    ROCPRIM_HOST_DEVICE constexpr inline custom_type() : x(NonZero ? 12 : 0), y(NonZero ? 34 : 0)
    {}

    ROCPRIM_HOST_DEVICE inline custom_type(T x, U y) : x(x), y(y) {}

    ROCPRIM_HOST_DEVICE inline custom_type(T xy) : x(xy), y(xy) {}

    template<typename V, typename W>
    ROCPRIM_HOST_DEVICE inline custom_type(const custom_type<V, W, NonZero>& other)
        : x(static_cast<T>(other.x)), y(static_cast<U>(other.y))
    {}

    ROCPRIM_HOST_DEVICE inline ~custom_type() = default;

    ROCPRIM_HOST_DEVICE
    inline custom_type
        operator+(const custom_type& other) const
    {
        rocprim::plus<T> plus_T;
        rocprim::plus<U> plus_U;
        return custom_type{plus_T(x, other.x), plus_U(y, other.y)};
    }

    ROCPRIM_HOST_DEVICE
    inline custom_type
        operator-(const custom_type& other) const
    {
        rocprim::minus<T> minus_T;
        rocprim::minus<U> minus_U;
        return custom_type(minus_T(x, other.x), minus_U(y, other.y));
    }

    ROCPRIM_HOST_DEVICE
    inline custom_type&
        operator=(const custom_type& other)
    {
        x = other.x;
        y = other.y;
        return *this;
    }

    ROCPRIM_HOST_DEVICE
    inline custom_type&
        operator+=(const custom_type& other)
    {
        x += other.x;
        y += other.y;
        return *this;
    }

    ROCPRIM_HOST_DEVICE
    inline bool
        operator<(const custom_type& other) const
    {
        rocprim::less<T>     less_T;
        rocprim::equal_to<T> equal_to_T;
        rocprim::less<U>     less_U;
        return (less_T(x, other.x) || (equal_to_T(x, other.x) && less_U(y, other.y)));
    }

    ROCPRIM_HOST_DEVICE
    inline bool
        operator>(const custom_type& other) const
    {
        rocprim::greater<T>  greater_T;
        rocprim::equal_to<T> equal_to_T;
        rocprim::greater<U>  greater_U;
        return (greater_T(x, other.x) || (equal_to_T(x, other.x) && greater_U(y, other.y)));
    }

    ROCPRIM_HOST_DEVICE
    inline bool
        operator==(const custom_type& other) const
    {
        rocprim::equal_to<T> equal_to_T;
        rocprim::equal_to<U> equal_to_U;
        return (equal_to_T(x, other.x) && equal_to_U(y, other.y));
    }

    ROCPRIM_HOST_DEVICE
    inline bool
        operator!=(const custom_type& other) const
    {
        return !(*this == other);
    }

    friend inline std::ostream& operator<<(std::ostream& stream, const custom_type& value)
    {
        stream << "[" << value.x << "; " << value.y << "]";
        return stream;
    }
};

template<typename T, typename U = T, bool NonZero = false>
struct custom_type_copyable
{
    using first_type  = T;
    using second_type = U;

    using value_type = std::conditional_t<std::is_same<T, U>::value, T, void>;

    T x;
    U y;

    ROCPRIM_HOST_DEVICE constexpr inline custom_type_copyable()
        : x(NonZero ? 12 : 0), y(NonZero ? 34 : 0)
    {}

    ROCPRIM_HOST_DEVICE inline custom_type_copyable(T x, U y) : x(x), y(y) {}

    ROCPRIM_HOST_DEVICE inline custom_type_copyable(T xy) : x(xy), y(xy) {}

    template<typename V, typename W>
    ROCPRIM_HOST_DEVICE inline custom_type_copyable(
        const custom_type_copyable<V, W, NonZero>& other)
        : x(static_cast<T>(other.x)), y(static_cast<U>(other.y))
    {}

    ROCPRIM_HOST_DEVICE
    inline bool
        operator<(const custom_type_copyable& other) const
    {
        rocprim::less<T>     less_T;
        rocprim::equal_to<T> equal_to_T;
        rocprim::less<U>     less_U;
        return (less_T(x, other.x) || (equal_to_T(x, other.x) && less_U(y, other.y)));
    }

    ROCPRIM_HOST_DEVICE
    inline bool
        operator>(const custom_type_copyable& other) const
    {
        rocprim::greater<T>  greater_T;
        rocprim::equal_to<T> equal_to_T;
        rocprim::greater<U>  greater_U;
        return (greater_T(x, other.x) || (equal_to_T(x, other.x) && greater_U(y, other.y)));
    }

    ROCPRIM_HOST_DEVICE
    inline bool
        operator==(const custom_type_copyable& other) const
    {
        rocprim::equal_to<T> equal_to_T;
        rocprim::equal_to<U> equal_to_U;
        return (equal_to_T(x, other.x) && equal_to_U(y, other.y));
    }

    ROCPRIM_HOST_DEVICE
    inline bool
        operator!=(const custom_type_copyable& other) const
    {
        return !(*this == other);
    }

    friend inline std::ostream& operator<<(std::ostream& stream, const custom_type_copyable& value)
    {
        stream << "[" << value.x << "; " << value.y << "]";
        return stream;
    }
};

static_assert(std::is_trivially_copyable<custom_type_copyable<char, double>>::value,
              "custom_type_copyable is not trivially copyable");

template<typename T>
struct is_custom_type_copyable : std::false_type
{};

template<typename T, typename U, bool NonZero>
struct is_custom_type_copyable<common::custom_type_copyable<T, U, NonZero>> : std::true_type
{};

template<unsigned int Size, class T, typename U = T, bool NonZero = false>
struct custom_huge_type : custom_type<T, U, NonZero>
{
    static constexpr auto extra_bytes = Size - sizeof(T) - sizeof(U);
    std::uint8_t          data[extra_bytes];

    // Non-zero values in default constructor for checking reduce and scan:
    // ensure that scan_op(custom_type(), value) != value
    ROCPRIM_HOST_DEVICE constexpr inline custom_huge_type() : custom_type<T, U, NonZero>()
    {}

    ROCPRIM_HOST_DEVICE inline custom_huge_type(T x, U y) : custom_type<T, U, NonZero>(x, y) {}

    ROCPRIM_HOST_DEVICE inline custom_huge_type(T xy) : custom_type<T, U, NonZero>(xy) {}

    template<typename V, typename W>
    ROCPRIM_HOST_DEVICE inline custom_huge_type(const custom_type<V, W, NonZero>& other)
        : custom_type<T, U, NonZero>(other)
    {}

    template<unsigned int OtherSize, typename V, typename W>
    ROCPRIM_HOST_DEVICE inline custom_huge_type(
        const custom_huge_type<OtherSize, V, W, NonZero>& other)
        : custom_type<T, U, NonZero>(static_cast<T>(other.x), static_cast<U>(other.y))
    {}

    friend inline std::ostream& operator<<(std::ostream& stream, const custom_huge_type& value)
    {
        stream << "[" << value.x << "; " << value.y << "]";
        return stream;
    }
};

template<typename T>
struct is_custom_type : std::false_type
{};

template<typename T, typename U, bool NonZero>
struct is_custom_type<common::custom_type<T, U, NonZero>> : std::true_type
{};

template<unsigned int Size, typename T, typename U, bool NonZero>
struct is_custom_type<common::custom_huge_type<Size, T, U, NonZero>> : std::true_type
{};

template<typename T, typename U, bool NonZero>
struct is_custom_type<common::custom_type_copyable<T, U, NonZero>> : std::true_type
{};
} // namespace common

#endif // COMMON_UTILS_CUSTOM_TYPE_HPP_

// Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_TEST_UTILS_CUSTOM_FLOAT_TRAITS_TYPE_HPP_
#define ROCPRIM_TEST_UTILS_CUSTOM_FLOAT_TRAITS_TYPE_HPP_

#include "test_utils_custom_test_types.hpp"

// For radix_key_codec
#include <rocprim/thread/radix_key_codec.hpp>

#include <ostream>
#include <type_traits>

#include <cmath>

namespace test_utils
{
// Custom type to model types like Eigen::half or Eigen::bfloat16, that wrap around floating point
// types.
struct custom_float_traits_type
{
    using value_type = float;
    float x;

    // Constructor for the data generation utilities, simply ignore the second number
    ROCPRIM_HOST_DEVICE custom_float_traits_type(float val, float /*ignored*/) : x{val}
    {}

    ROCPRIM_HOST_DEVICE custom_float_traits_type(float val) : x{val} {}

    custom_float_traits_type() = default;

    ROCPRIM_HOST_DEVICE
    custom_float_traits_type
        operator+(const custom_float_traits_type& other) const
    {
        return custom_float_traits_type(x + other.x);
    }

    ROCPRIM_HOST_DEVICE
    custom_float_traits_type
        operator-(const custom_float_traits_type& other) const
    {
        return custom_float_traits_type(x - other.x);
    }

    ROCPRIM_HOST_DEVICE
    bool operator<(const custom_float_traits_type& other) const
    {
        return x < other.x;
    }

    ROCPRIM_HOST_DEVICE
    bool operator>(const custom_float_traits_type& other) const
    {
        return x > other.x;
    }

    ROCPRIM_HOST_DEVICE
    bool operator==(const custom_float_traits_type& other) const
    {
        return x == other.x;
    }

    ROCPRIM_HOST_DEVICE
    bool operator!=(const custom_float_traits_type& other) const
    {
        return !(*this == other);
    }
};

inline bool signbit(const custom_float_traits_type& val)
{
    return std::signbit(val.x);
}

inline std::ostream& operator<<(std::ostream& stream, const custom_float_traits_type& value)
{
    stream << "[" << value.x << "]";
    return stream;
}

template<>
struct is_custom_test_type<custom_float_traits_type> : std::true_type
{};

template<>
struct inner_type<custom_float_traits_type>
{
    using type = custom_float_traits_type::value_type;
};

} // namespace test_utils

template<>
struct ::rocprim::traits::define<test_utils::custom_float_traits_type>
{
    using is_arithmetic = ::rocprim::traits::is_arithmetic::values<true>;
    using number_format = ::rocprim::traits::number_format::values<
        ::rocprim::traits::number_format::kind::floating_point_type>;
    using float_bit_mask
        = ::rocprim::traits::float_bit_mask::values<uint32_t, 0x80000000, 0x7F800000, 0x007FFFFF>;
};

template<>
struct ::rocprim::detail::radix_key_codec_base<test_utils::custom_float_traits_type>
    : ::rocprim::detail::radix_key_codec_floating<test_utils::custom_float_traits_type,
                                                  unsigned int>
{};

#endif //ROCPRIM_TEST_UTILS_CUSTOM_FLOAT_TYPE_HPP_

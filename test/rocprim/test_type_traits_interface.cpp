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

#include "../common_test_header.hpp"

#include "test_utils_custom_float_type.hpp"

#include <rocprim/config.hpp>
#include <rocprim/type_traits.hpp>
#include <rocprim/types.hpp>

#include <ostream>
#include <stdint.h>

#define ROCPRIM_STATIC_ASSERT(cond) static_assert((cond), "Rocprim Traits Assertion failed!")
#define ROCPRIM_STATIC_ASSERT_TRUE(cond) ROCPRIM_STATIC_ASSERT((cond))
#define ROCPRIM_STATIC_ASSERT_FALSE(cond) ROCPRIM_STATIC_ASSERT(!(cond))
#define ROCPRIM_STATIC_ASSERT_EQ(val1, val2) ROCPRIM_STATIC_ASSERT((val1) == (val2))
#define ROCPRIM_STATIC_ASSERT_NE(val1, val2) ROCPRIM_STATIC_ASSERT((val1) != (val2))

namespace type_traits_test
{
// Custom type to model types like Eigen::half or Eigen::bfloat16, that wrap around floating point
// types.
struct custom_float_type : test_utils::custom_float_type
{};

inline bool signbit(const custom_float_type& val)
{
    return test_utils::signbit(val);
}

inline std::ostream& operator<<(std::ostream& stream, const custom_float_type& value)
{
    return test_utils::operator<<(stream, value);
}

// Custom type to model types like Eigen::half or Eigen::bfloat16, that wrap around floating point
// types.
struct custom_int_type
{
    int x;

    // Constructor for the data generation utilities, simply ignore the second number
    ROCPRIM_HOST_DEVICE custom_int_type(int val, int /*ignored*/) : x{val}
    {}

    ROCPRIM_HOST_DEVICE custom_int_type(int val) : x{val} {}

    custom_int_type() = default;

    ROCPRIM_HOST_DEVICE
    custom_int_type
        operator+(const custom_int_type& other) const
    {
        return custom_int_type(x + other.x);
    }

    ROCPRIM_HOST_DEVICE
    custom_int_type
        operator-(const custom_int_type& other) const
    {
        return custom_int_type(x - other.x);
    }

    ROCPRIM_HOST_DEVICE
    bool operator<(const custom_int_type& other) const
    {
        return x < other.x;
    }

    ROCPRIM_HOST_DEVICE
    bool operator>(const custom_int_type& other) const
    {
        return x > other.x;
    }

    ROCPRIM_HOST_DEVICE
    bool operator==(const custom_int_type& other) const
    {
        return x == other.x;
    }

    ROCPRIM_HOST_DEVICE
    bool operator!=(const custom_int_type& other) const
    {
        return !(*this == other);
    }
};

inline std::ostream& operator<<(std::ostream& stream, const custom_int_type& value)
{
    stream << "[" << value.x << "]";
    return stream;
}

} // namespace type_traits_test

template<>
struct rocprim::traits::define<type_traits_test::custom_float_type>
{
    using is_arithmetic = rocprim::traits::is_arithmetic::values<true>;
    using number_format
        = rocprim::traits::number_format::values<number_format::kind::floating_point_type>;
    using float_bit_mask = rocprim::traits::float_bit_mask::values<uint32_t, 10, 10, 10>;
};

template<>
struct rocprim::traits::define<type_traits_test::custom_int_type>
{
    using is_arithmetic = rocprim::traits::is_arithmetic::values<true>;
    using number_format
        = rocprim::traits::number_format::values<number_format::kind::integral_type>;
    using integral_sign
        = rocprim::traits::integral_sign::values<traits::integral_sign::kind::signed_type>;
};

template<class Params>
class RocprimFloatingPointTests : public ::testing::Test
{
public:
    using input_type = Params;
};
using FloatingPointTypeTestParams = ::testing::
    Types<rocprim::half, rocprim::bfloat16, float, double, type_traits_test::custom_float_type>;

template<class Params>
class RocprimIntegralTests : public ::testing::Test
{
public:
    using input_type = Params;
};
using IntegralTypeTestParams = ::testing::Types<uint8_t,
                                                int8_t,
                                                uint16_t,
                                                int16_t,
                                                uint32_t,
                                                int32_t,
                                                uint64_t,
                                                int64_t,
                                                rocprim::uint128_t,
                                                rocprim::int128_t,
                                                type_traits_test::custom_int_type>;

TYPED_TEST_SUITE(RocprimFloatingPointTests, FloatingPointTypeTestParams);

TYPED_TEST_SUITE(RocprimIntegralTests, IntegralTypeTestParams);

TYPED_TEST(RocprimFloatingPointTests, FloatingPoint)
{
    using input_type            = typename TestFixture::input_type;
    constexpr auto input_traits = rocprim::traits::get<input_type>();

    ROCPRIM_STATIC_ASSERT_TRUE(input_traits.is_arithmetic());
    ROCPRIM_STATIC_ASSERT_TRUE(input_traits.is_fundamental());
    ROCPRIM_STATIC_ASSERT_FALSE(input_traits.is_compound());
    ROCPRIM_STATIC_ASSERT_TRUE(input_traits.is_scalar());
    ROCPRIM_STATIC_ASSERT_TRUE(input_traits.is_floating_point());
    ROCPRIM_STATIC_ASSERT_FALSE(input_traits.is_integral());

    ROCPRIM_STATIC_ASSERT_EQ(input_traits.is_integral(), rocprim::is_integral<input_type>::value);

    // cannot do static_assert because under c++ 14 there is no if constexpr
    if constexpr(rocprim::is_arithmetic<input_type>::value)
    { // for c++ arithmetic types
        ASSERT_EQ(input_traits.is_compound(), rocprim::is_compound<input_type>::value);
        ASSERT_EQ(input_traits.is_scalar(), rocprim::is_scalar<input_type>::value);
        ASSERT_EQ(input_traits.is_fundamental(), rocprim::is_fundamental<input_type>::value);
        ASSERT_EQ(input_traits.is_arithmetic(), rocprim::is_arithmetic<input_type>::value);
        ASSERT_EQ(input_traits.is_floating_point(), rocprim::is_floating_point<input_type>::value);
    }
    else
    { // for custom_types
        ASSERT_NE(input_traits.is_compound(), rocprim::is_compound<input_type>::value);
        ASSERT_NE(input_traits.is_scalar(), rocprim::is_scalar<input_type>::value);
        ASSERT_NE(input_traits.is_fundamental(), rocprim::is_fundamental<input_type>::value);
        ASSERT_NE(input_traits.is_arithmetic(), rocprim::is_arithmetic<input_type>::value);
        ASSERT_NE(input_traits.is_floating_point(), rocprim::is_floating_point<input_type>::value);
    }

    [[maybe_unused]] constexpr auto float_bit_mask = input_traits.float_bit_mask();
}

TYPED_TEST(RocprimIntegralTests, Integral)
{
    using input_type = typename TestFixture::input_type;

    constexpr auto input_traits = rocprim::traits::get<input_type>();

    ROCPRIM_STATIC_ASSERT_TRUE(input_traits.is_arithmetic());
    ROCPRIM_STATIC_ASSERT_TRUE(input_traits.is_fundamental());
    ROCPRIM_STATIC_ASSERT_FALSE(input_traits.is_compound());
    ROCPRIM_STATIC_ASSERT_TRUE(input_traits.is_scalar());
    ROCPRIM_STATIC_ASSERT_FALSE(input_traits.is_floating_point());
    ROCPRIM_STATIC_ASSERT_TRUE(input_traits.is_integral());

    ROCPRIM_STATIC_ASSERT_EQ(input_traits.is_floating_point(),
                             rocprim::is_floating_point<input_type>::value);
    ROCPRIM_STATIC_ASSERT_NE(input_traits.is_signed(), input_traits.is_unsigned());

    if constexpr(rocprim::is_arithmetic<input_type>::value)
    { // for c++ arithmetic types
        ASSERT_EQ(input_traits.is_compound(), rocprim::is_compound<input_type>::value);
        ASSERT_EQ(input_traits.is_scalar(), rocprim::is_scalar<input_type>::value);
        ASSERT_EQ(input_traits.is_fundamental(), rocprim::is_fundamental<input_type>::value);
        ASSERT_EQ(input_traits.is_arithmetic(), rocprim::is_arithmetic<input_type>::value);
        ASSERT_EQ(input_traits.is_integral(), rocprim::is_integral<input_type>::value);
        ASSERT_EQ(input_traits.is_signed(), rocprim::is_signed<input_type>::value);
        ASSERT_EQ(input_traits.is_unsigned(), rocprim::is_unsigned<input_type>::value);
    }
    else
    { // for custom_types
        ASSERT_NE(input_traits.is_compound(), rocprim::is_compound<input_type>::value);
        ASSERT_NE(input_traits.is_scalar(), rocprim::is_scalar<input_type>::value);
        ASSERT_NE(input_traits.is_fundamental(), rocprim::is_fundamental<input_type>::value);
        ASSERT_NE(input_traits.is_arithmetic(), rocprim::is_arithmetic<input_type>::value);
        ASSERT_NE(input_traits.is_integral(), rocprim::is_integral<input_type>::value);
    }
}

TEST(TraitsInterface, AllTypes)
{
    struct TestT
    {};
    constexpr auto input_traits = rocprim::traits::get<TestT>();
    ROCPRIM_STATIC_ASSERT_FALSE(input_traits.is_arithmetic());
    ROCPRIM_STATIC_ASSERT_FALSE(input_traits.is_fundamental());
    ROCPRIM_STATIC_ASSERT_TRUE(input_traits.is_compound());
    ROCPRIM_STATIC_ASSERT_FALSE(input_traits.is_scalar());
    ROCPRIM_STATIC_ASSERT_FALSE(input_traits.is_floating_point());
    ROCPRIM_STATIC_ASSERT_FALSE(input_traits.is_integral());
}

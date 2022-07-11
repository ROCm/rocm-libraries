#ifndef ROCPRIM_TEST_UTILS_CUSTOM_FLOAT_TYPE_HPP_
#define ROCPRIM_TEST_UTILS_CUSTOM_FLOAT_TYPE_HPP_

#include "test_utils_custom_test_types.hpp"

// For radix_key_codec
#include <rocprim/detail/radix_sort.hpp>

#include <ostream>
#include <type_traits>

#include <cmath>

namespace test_utils
{
// Custom type to model types like Eigen::half or Eigen::bfloat16, that wrap around floating point
// types.
struct custom_float_type
{
    using value_type = float;
    float x;

    // Constructor for the data generation utilities, simply ignore the second number
    ROCPRIM_HOST_DEVICE
    custom_float_type(float val, float /*ignored*/) : x{val} {}

    ROCPRIM_HOST_DEVICE
    custom_float_type(float val) : x{val} {}

    custom_float_type() = default;

    ROCPRIM_HOST_DEVICE custom_float_type operator+(const custom_float_type& other) const
    {
        return custom_float_type(x + other.x);
    }

    ROCPRIM_HOST_DEVICE custom_float_type operator-(const custom_float_type& other) const
    {
        return custom_float_type(x - other.x);
    }

    ROCPRIM_HOST_DEVICE bool operator<(const custom_float_type& other) const
    {
        return x < other.x;
    }

    ROCPRIM_HOST_DEVICE bool operator>(const custom_float_type& other) const
    {
        return x > other.x;
    }

    ROCPRIM_HOST_DEVICE bool operator==(const custom_float_type& other) const
    {
        return x == other.x;
    }

    ROCPRIM_HOST_DEVICE bool operator!=(const custom_float_type& other) const
    {
        return !(*this == other);
    }
};

inline bool signbit(const custom_float_type& val)
{
    return std::signbit(val.x);
}

inline std::ostream& operator<<(std::ostream& stream, const custom_float_type& value)
{
    stream << "[" << value.x << "]";
    return stream;
}

template<>
struct is_custom_test_type<custom_float_type> : std::true_type
{};

template<>
struct inner_type<custom_float_type>
{
    using type = custom_float_type::value_type;
};

} // namespace test_utils

// This is how libraries "hack" rocprim to accept "custom" floating point types in radix based sorts
// because this is something that is unavoidable in some cases we should provide clear customization
// points instead of hacks like these.
// Nonetheless until that adding a test for this pattern should reduce accidental breakages
namespace rocprim
{
namespace detail
{

template<>
struct float_bit_mask<test_utils::custom_float_type>
{
    static constexpr uint32_t sign_bit = 0x80000000;
    static constexpr uint32_t exponent = 0x7F800000;
    static constexpr uint32_t mantissa = 0x007FFFFF;
    using bit_type                     = uint32_t;
};

template<>
struct radix_key_codec_base<test_utils::custom_float_type>
    : radix_key_codec_floating<test_utils::custom_float_type, unsigned int>
{};

} // namespace detail
} // namespace rocprim

#endif //ROCPRIM_TEST_UTILS_CUSTOM_FLOAT_TYPE_HPP_

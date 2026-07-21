// MIT License
//
// Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPCUB_TEST_HIPCUB_TEST_UTILS_DATA_GENERATION_HPP_
#define HIPCUB_TEST_HIPCUB_TEST_UTILS_DATA_GENERATION_HPP_

// Std::memcpy and std::memcmp
#include <cstring>

#include <type_traits>

#include "test_utils_bfloat16.hpp"
#include "test_utils_custom_test_types.hpp"
#include "test_utils_half.hpp"

namespace test_utils
{

template<typename T>
HIPCUB_HOST_DEVICE
T set_half_bits(uint16_t value)
{
    T              half_value{};
    unsigned char* char_representation = reinterpret_cast<unsigned char*>(&half_value);
    char_representation[0]             = value;
    char_representation[1]             = value >> 8;
    return half_value;
}

// Numeric limits which also supports custom_test_type<U> classes
template<class T>
struct numeric_limits : std::numeric_limits<T>
{};

template<>
struct numeric_limits<test_utils::half> : public std::numeric_limits<test_utils::half>
{
public:
    using T = test_utils::half;
    static inline T min()
    {
        return T(0.00006104f);
    };
    static inline T max()
    {
        return set_half_bits<T>(0x7bff);
    };
    static inline T lowest()
    {
        return set_half_bits<T>(0xfbff);
    };
    static inline T infinity()
    {
        return set_half_bits<T>(0x7c00);
    };
    static inline T quiet_NaN()
    {
        return T(std::numeric_limits<float>::quiet_NaN());
    };
    static inline T signaling_NaN()
    {
        return T(std::numeric_limits<float>::signaling_NaN());
    };
    static inline T infinity_neg()
    {
        return set_half_bits<T>(0xfc00);
    };
};

template<>
class numeric_limits<test_utils::bfloat16> : public std::numeric_limits<test_utils::bfloat16>
{
public:
    using T = test_utils::bfloat16;
    static inline T max()
    {
        return set_half_bits<T>(0x7f7f);
    };
    static inline T min()
    {
        return T(std::numeric_limits<float>::min());
    };
    static inline T lowest()
    {
        return set_half_bits<T>(0xff7f);
    };
    static inline T infinity()
    {
        return set_half_bits<T>(0x7f80);
    };
    static inline T quiet_NaN()
    {
        return T(std::numeric_limits<float>::quiet_NaN());
    };
    static inline T signaling_NaN()
    {
        return T(std::numeric_limits<float>::signaling_NaN());
    };
    static inline T infinity_neg()
    {
        return set_half_bits<T>(0xff80);
    };
};

template<>
class numeric_limits<float> : public std::numeric_limits<float>
{
public:
    static inline float infinity_neg()
    {
        return -std::numeric_limits<float>::infinity();
    };
};
// End of extended numeric_limits

#if HIPCUB_IS_INT128_ENABLED
template<class T>
using is_int128 = std::is_same<__int128_t, typename std::remove_cv<T>::type>;
template<class T>
using is_uint128 = std::is_same<__uint128_t, typename std::remove_cv<T>::type>;
#else
template<class T>
using is_int128 = std::false_type;
template<class T>
using is_uint128 = std::false_type;
#endif // HIPCUB_IS_INT128_ENABLED

template<class T>
using is_half = std::is_same<test_utils::half, typename std::remove_cv<T>::type>;

template<class T>
using is_bfloat16 = std::is_same<test_utils::bfloat16, typename std::remove_cv<T>::type>;

template<class T>
using is_native_half = std::is_same<test_utils::native_half, typename std::remove_cv<T>::type>;

template<class T>
using is_native_bfloat16
    = std::is_same<test_utils::native_bfloat16, typename std::remove_cv<T>::type>;

template<class T>
struct convert_to_native_t_impl
{
    using type = T;
};

template<>
struct convert_to_native_t_impl<test_utils::bfloat16>
{
    using type = test_utils::native_bfloat16;
};
template<>
struct convert_to_native_t_impl<test_utils::half>
{
    using type = test_utils::native_half;
};

template<class T>
using convert_to_native_t = typename convert_to_native_t_impl<T>::type;

// is_floating_point which supports custom_test_type<U> classes
template<class T>
struct is_special_floating_point
    : std::integral_constant<bool,
                             is_half<T>::value || is_bfloat16<T>::value || is_native_half<T>::value
                                 || is_native_bfloat16<T>::value>
{};

template<class T>
struct is_floating_point
    : std::integral_constant<bool,
                             std::is_floating_point<T>::value
                                 || is_special_floating_point<T>::value>
{};

// is_integral which supports custom_test_type<U> classes
template<class T>
struct is_integral : std::integral_constant<bool, std::is_integral<T>::value>
{};

template<class T>
struct is_arithmetic
    : std::integral_constant<bool, is_integral<T>::value || is_floating_point<T>::value>
{};

// Converts possible device side types to their relevant host side native types
inline test_utils::native_half convert_to_native(test_utils::half value)
{
    return test_utils::native_half(value);
}

inline test_utils::native_bfloat16 convert_to_native(const test_utils::bfloat16& value)
{
    return test_utils::native_bfloat16(value);
}

template<class T>
inline auto convert_to_native(const T& value) ->
    typename std::enable_if<!(is_half<T>::value || is_bfloat16<T>::value), T>::type
{
    return value;
}

// Converts possible host side native types to their relevant device side types
template<class U, class T>
inline auto convert_to_device(const T& value) ->
    typename std::enable_if<is_half<U>::value, test_utils::half>::type
{
    return test_utils::native_to_half(value);
}

template<class U, class T>
inline auto convert_to_device(T value) ->
    typename std::enable_if<is_bfloat16<U>::value, test_utils::bfloat16>::type
{
#ifdef __HIP_PLATFORM_NVIDIA__
    // __nv__bfloat16 has no cast from int and gets confused wether to
    // cast via float or double.
    if(std::is_integral<T>::value)
    {
        return test_utils::native_to_bfloat16(static_cast<float>(value));
    }
#endif

    return test_utils::native_to_bfloat16(value);
}

template<class U, class T>
inline auto convert_to_device(T value) ->
    typename std::enable_if<!(is_half<U>::value || is_bfloat16<U>::value), U>::type
{
    return static_cast<U>(value);
}

template<class T>
using convert_to_fundamental_t
    = std::conditional_t<is_half<T>::value || is_bfloat16<T>::value, float, T>;

template<class T>
inline auto convert_to_fundamental(T value)
{
    return static_cast<convert_to_fundamental_t<T>>(value);
}

// Helper class to generate a vector of special values for any type
template<class T>
struct special_values
{
private:
    // sign_bit_flip needed because host-side operators for __half are missing. (e.g. -__half unary operator or (-1*) __half*__half binary operator
    static T sign_bit_flip(T value)
    {
        uint8_t* data = reinterpret_cast<uint8_t*>(&value);
        data[sizeof(T) - 1] ^= 0x80;
        return value;
    }

public:
    static std::vector<T> vector()
    {
        if(std::is_integral<T>::value)
        {
            return std::vector<T>();
        }
        else
        {
            using traits          = hipcub::NumericTraits<T>;
            using unsigned_bits   = typename traits::UnsignedBits;
            auto nan_with_payload = [](const unsigned_bits& payload)
            {
                T             value = test_utils::numeric_limits<T>::quiet_NaN();
                unsigned_bits int_value;
                std::memcpy(&int_value, &value, sizeof(T));
                int_value |= payload;
                std::memcpy(&value, &int_value, sizeof(T));
                return value;
            };

            std::vector<T> r = {
                test_utils::numeric_limits<T>::quiet_NaN(),
                sign_bit_flip(test_utils::numeric_limits<T>::quiet_NaN()),
                //test_utils::numeric_limits<T>::signaling_NaN(), // signaling_NaN not supported on NVIDIA yet
                //sign_bit_flip(test_utils::numeric_limits<T>::signaling_NaN()),
                test_utils::numeric_limits<T>::infinity(),
                sign_bit_flip(test_utils::numeric_limits<T>::infinity()),
                T(0.0),
                T(-0.0),
                nan_with_payload(0x11),
                nan_with_payload(0x80),
                nan_with_payload(0x1)};
            return r;
        }
    }
};
// end of special_values helpers

/// Insert special values of type T at a random place in the source vector
/// \tparam T
/// \param source The source vector<T> to modify
template<class T>
void add_special_values(std::vector<T>& source, int seed_value)
{
    std::default_random_engine gen(seed_value);
    std::vector<T>             special_values = test_utils::special_values<T>::vector();
    if(source.size() > special_values.size())
    {
        unsigned int start = gen() % (source.size() - special_values.size());
        std::copy(special_values.begin(), special_values.end(), source.begin() + start);
    }
}

// std::uniform_int_distribution is undefined for anything other than
// short, int, long, long long, unsigned short, unsigned int, unsigned long, or unsigned long long.
// Actually causes problems with signed/unsigned char on Windows using clang.
template<typename T>
struct is_valid_for_int_distribution
    : std::integral_constant<
          bool,
          std::is_same<short, T>::value || std::is_same<unsigned short, T>::value
              || std::is_same<int, T>::value || std::is_same<unsigned int, T>::value
              || std::is_same<long, T>::value || std::is_same<unsigned long, T>::value
              || std::is_same<long long, T>::value || std::is_same<unsigned long long, T>::value>
{};

template<class T>
inline auto get_random_data(size_t size, T min, T max, int seed_value) ->
    typename std::enable_if<std::is_integral<T>::value, std::vector<T>>::type
{
    std::default_random_engine gen(seed_value);
    using dis_type = typename std::conditional<
        is_valid_for_int_distribution<T>::value,
        T,
        typename std::conditional<std::is_signed<T>::value, int, unsigned int>::type>::type;
    std::uniform_int_distribution<dis_type> distribution(static_cast<dis_type>(min),
                                                         static_cast<dis_type>(max));
    std::vector<T>                          data(size);
    std::generate(data.begin(), data.end(), [&]() { return distribution(gen); });
    return data;
}

template<class T, class S, class U>
inline auto get_random_data(size_t size, S min, U max, int seed_value) ->
    typename std::enable_if<!std::is_integral<T>::value && !is_custom_test_type<T>::value
                                && !std::is_same<T, __int128_t>::value
                                && !std::is_same<T, __uint128_t>::value,
                            std::vector<T>>::type
{
    std::default_random_engine gen(seed_value);
    using dis_type =
        typename std::conditional<test_utils::is_special_floating_point<T>::value, float, T>::type;
    std::uniform_real_distribution<dis_type> distribution(static_cast<dis_type>(min),
                                                          static_cast<dis_type>(max));
    std::vector<T>                           data(size);
    std::generate(data.begin(), data.end(), [&]() { return static_cast<T>(distribution(gen)); });
    return data;
}

template<class T, class S, class U>
inline auto get_random_data(size_t size, S min, U max, int seed_value) ->
    typename std::enable_if<std::is_same<T, __int128_t>::value, std::vector<T>>::type
{
    std::default_random_engine             gen(seed_value);
    std::uniform_int_distribution<int64_t> distribution(static_cast<int64_t>(min),
                                                        static_cast<int64_t>(max));
    std::vector<T>                         data(size);
    std::generate(data.begin(),
                  data.end(),
                  [&]() { return static_cast<T>(distribution(gen)) * 2; });
    return data;
}

template<class T, class S, class U>
inline auto get_random_data(size_t size, S min, U max, int seed_value) ->
    typename std::enable_if<std::is_same<T, __uint128_t>::value, std::vector<T>>::type
{
    std::default_random_engine              gen(seed_value);
    std::uniform_int_distribution<uint64_t> distribution(static_cast<uint64_t>(min),
                                                         static_cast<uint64_t>(max));
    std::vector<T>                          data(size);
    std::generate(data.begin(),
                  data.end(),
                  [&]() { return static_cast<T>(distribution(gen)) * 2; });
    return data;
}

template<class T>
inline auto get_random_data(size_t                 size,
                            typename T::value_type min,
                            typename T::value_type max,
                            int                    seed_value) ->
    typename std::enable_if<is_custom_test_type<T>::value
                                && std::is_integral<typename T::value_type>::value,
                            std::vector<T>>::type
{
    std::default_random_engine gen(seed_value);
    using dis_type = typename std::conditional<
        is_valid_for_int_distribution<typename T::value_type>::value,
        typename T::value_type,
        typename std::conditional<std::is_signed<typename T::value_type>::value,
                                  int,
                                  unsigned int>::type>::type;
    std::uniform_int_distribution<dis_type> distribution(static_cast<dis_type>(min),
                                                         static_cast<dis_type>(max));
    std::vector<T>                          data(size);
    std::generate(data.begin(),
                  data.end(),
                  [&]() { return T(distribution(gen), distribution(gen)); });
    return data;
}

template<class T>
inline auto get_random_data(size_t                 size,
                            typename T::value_type min,
                            typename T::value_type max,
                            int                    seed_value) ->
    typename std::enable_if<is_custom_test_type<T>::value
                                && std::is_floating_point<typename T::value_type>::value,
                            std::vector<T>>::type
{
    std::default_random_engine                             gen(seed_value);
    std::uniform_real_distribution<typename T::value_type> distribution(min, max);
    std::vector<T>                                         data(size);
    std::generate(data.begin(),
                  data.end(),
                  [&]() { return T(distribution(gen), distribution(gen)); });
    return data;
}

template<class T>
inline auto get_random_value(T min, T max, int seed_value) ->
    typename std::enable_if<test_utils::is_arithmetic<T>::value, T>::type
{
    return get_random_data<T>(1, min, max, seed_value)[0];
}

template<class T>
inline std::vector<T> get_random_data01(size_t size, float p, int seed_value)
{
    const size_t                max_random_size = 1024 * 1024;
    std::default_random_engine  gen(seed_value);
    std::bernoulli_distribution distribution(p);
    std::vector<T>              data(size);
    std::generate(data.begin(),
                  data.begin() + std::min(size, max_random_size),
                  [&]() { return convert_to_device<T>(distribution(gen)); });
    for(size_t i = max_random_size; i < size; i += max_random_size)
    {
        std::copy_n(data.begin(), std::min(size - i, max_random_size), data.begin() + i);
    }
    return data;
}

template<unsigned int MaxPow2 = 35>
inline std::vector<size_t> get_large_sizes(int seed_value)
{
    // clang-format off
    std::vector<size_t> sizes = {
        (size_t{1} << 32) - 1, size_t{1} << 32,
        (size_t{1} << MaxPow2) - 1, size_t{1} << MaxPow2
    };
    // clang-format on
    const std::vector<size_t> random_sizes
        = test_utils::get_random_data<size_t>(2,
                                              (size_t{1} << 30) + 1,
                                              (size_t{1} << MaxPow2) - 2,
                                              seed_value);
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    std::sort(sizes.begin(), sizes.end());
    return sizes;
}

template<class T>
inline std::vector<size_t> get_sizes(T seed_value)
{
    // clang-format off
    std::vector<size_t> sizes = {
        1024, 2048, 4096, 1792,
        1, 10, 53, 211, 500, 2345,
        11001, 34567, 100000,
        (1 << 16) - 1220,
        (1 << 20) + 123
    };
    // clang-format on
    const std::vector<size_t> random_sizes
        = test_utils::get_random_data<size_t>(10, 1, 100000, seed_value);
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    return sizes;
}

} // namespace test_utils

#endif // HIPCUB_TEST_HIPCUB_TEST_UTILS_DATA_GENERATION_HPP_

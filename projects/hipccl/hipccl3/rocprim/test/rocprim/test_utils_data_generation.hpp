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

#ifndef ROCPRIM_TEST_UTILS_DATA_GENERATION_HPP
#define ROCPRIM_TEST_UTILS_DATA_GENERATION_HPP

// Std::memcpy and std::memcmp
#include <cstring>

#include "test_utils_half.hpp"
#include "test_utils_bfloat16.hpp"
#include "test_utils_custom_test_types.hpp"

namespace test_utils {

static constexpr uint32_t random_data_generation_segments = 32;
static constexpr uint32_t random_data_generation_repeat_strides = 4;

// std::uniform_int_distribution is undefined for anything other than
// short, int, long, long long, unsigned short, unsigned int, unsigned long, or unsigned long long.
template <typename T>
struct is_valid_for_int_distribution :
    std::integral_constant<bool,
                           std::is_same<short, T>::value ||
                               std::is_same<unsigned short, T>::value ||
                               std::is_same<int, T>::value ||
                               std::is_same<unsigned int, T>::value ||
                               std::is_same<long, T>::value ||
                               std::is_same<unsigned long, T>::value ||
                               std::is_same<long long, T>::value ||
                               std::is_same<unsigned long long, T>::value
                           > {};
namespace detail
{
    template<class T>
    struct numeric_limits_custom_test_type : public std::numeric_limits<typename T::value_type>
    {
    };
}

// Numeric limits which also supports custom_test_type<U> classes
template<class T>
struct numeric_limits : public std::conditional<
                            is_custom_test_type<T>::value,
                            detail::numeric_limits_custom_test_type<T>,
                            std::numeric_limits<T>
                            >::type
{
};

template<> struct numeric_limits<test_utils::half> : public std::numeric_limits<test_utils::half> {
public:
    using T = test_utils::half;
    static inline T min() {
        return T(0.00006104f);
    };
    static inline T max() {
        return T(65504.0f);
    };
    static inline T lowest() {
        return T(-65504.0f);
    };
    static inline T infinity() {
        return T(std::numeric_limits<float>::infinity());
    };
    static inline T quiet_NaN() {
        return T(std::numeric_limits<float>::quiet_NaN());
    };
    static inline T signaling_NaN() {
        return T(std::numeric_limits<float>::signaling_NaN());
    };
};

template<> class numeric_limits<test_utils::bfloat16> : public std::numeric_limits<test_utils::bfloat16> {
public:
    using T = test_utils::bfloat16;

    static inline T max() {
        return T(std::numeric_limits<float>::max()*0.998);
    };
    static inline T min() {
        return T(std::numeric_limits<float>::min());
    };
    static inline T lowest() {
        return T(std::numeric_limits<float>::lowest()*0.998);
    };
    static inline T infinity() {
        return T(std::numeric_limits<float>::infinity());
    };
    static inline T quiet_NaN() {
        return T(std::numeric_limits<float>::quiet_NaN());
    };
    static inline T signaling_NaN() {
        return T(std::numeric_limits<float>::signaling_NaN());
    };
};
// End of extended numeric_limits

// Helper class to generate a vector of special values for any type
template<class T>
struct special_values {
private:
    // sign_bit_flip needed because host-side operators for __half are missing. (e.g. -__half unary operator or (-1*) __half*__half binary operator
    static T sign_bit_flip(T value){
        uint8_t* data = reinterpret_cast<uint8_t*>(&value);
        data[sizeof(T)-1] ^= 0x80;
        return value;
    }

public:
    static std::vector<T> vector(){
        if(std::is_integral<T>::value){
            return std::vector<T>();
        }else {
            std::vector<T> r = {test_utils::numeric_limits<T>::quiet_NaN(),
                                sign_bit_flip(test_utils::numeric_limits<T>::quiet_NaN()),
                                // TODO: switch on when signaling_NaN will be supported on NVIDIA
                                //test_utils::numeric_limits<T>::signaling_NaN(),
                                //sign_bit_flip(test_utils::numeric_limits<T>::signaling_NaN()),
                                test_utils::numeric_limits<T>::infinity(),
                                sign_bit_flip(test_utils::numeric_limits<T>::infinity()),
                                T(0.0),
                                T(-0.0)};
            return r;
        }
    }
};
// end of special_values helpers

/// Insert special values of type T at a random place in the source vector
/// \tparam T
/// \param source The source vector<T> to modify
template<class T>
void add_special_values(std::vector<T>& source, seed_type seed_value)
{
    engine_type gen{seed_value};
    std::vector<T> special_values = test_utils::special_values<T>::vector();
    if(source.size() > special_values.size())
    {
        unsigned int start = gen() % (source.size() - special_values.size());
        std::copy(special_values.begin(), special_values.end(), source.begin() + start);
    }
}

template<class T, class U, class V>
inline auto get_random_data(size_t size, U min, V max, seed_type seed_value)
    -> typename std::enable_if<rocprim::is_integral<T>::value, std::vector<T>>::type
{
    engine_type gen{seed_value};
    using dis_type = typename std::conditional<
        is_valid_for_int_distribution<T>::value,
        T,
        typename std::conditional<std::is_signed<T>::value,
                                  int,
                                  unsigned int>::type
        >::type;
    std::uniform_int_distribution<dis_type> distribution(static_cast<dis_type>(min), static_cast<dis_type>(max));
    std::vector<T> data(size);
    size_t segment_size = size / random_data_generation_segments;
    if(segment_size != 0)
    {
        for(uint32_t segment_index = 0; segment_index < random_data_generation_segments; segment_index++)
        {
            if(segment_index % random_data_generation_repeat_strides == 0)
            {
                T repeated_value = static_cast<T>(distribution(gen));
                std::fill(
                    data.begin() + segment_size * segment_index,
                    data.begin() + segment_size * (segment_index + 1),
                    repeated_value);

            }
            else
            {
                std::generate(
                    data.begin() + segment_size * segment_index,
                    data.begin() + segment_size * (segment_index + 1),
                    [&]() { return static_cast<T>(distribution(gen)); });
            }
        }
    }
    else
    {
        std::generate(data.begin(), data.end(), [&]() { return static_cast<T>(distribution(gen)); });
    }
    return data;
}

template<class T, class U, class V>
inline auto get_random_data(size_t size, U min, V max, seed_type seed_value)
    -> typename std::enable_if<rocprim::is_floating_point<T>::value, std::vector<T>>::type
{
    engine_type gen{seed_value};
    // Generate floats when T is half or bfloat16
    using dis_type = typename std::conditional<std::is_same<rocprim::half, T>::value || std::is_same<rocprim::bfloat16, T>::value, float, T>::type;
    std::uniform_real_distribution<dis_type> distribution(static_cast<dis_type>(min), static_cast<dis_type>(max));
    std::vector<T> data(size);
    size_t segment_size = size / random_data_generation_segments;
    if(segment_size != 0)
    {
        for(uint32_t segment_index = 0; segment_index < random_data_generation_segments; segment_index++)
        {
            if(segment_index % random_data_generation_repeat_strides == 0)
            {
                T repeated_value = static_cast<T>(distribution(gen));
                std::fill(
                    data.begin() + segment_size * segment_index,
                    data.begin() + segment_size * (segment_index + 1),
                    repeated_value);

            }
            else
            {
                std::generate(
                    data.begin() + segment_size * segment_index,
                    data.begin() + segment_size * (segment_index + 1),
                    [&]() { return static_cast<T>(distribution(gen)); });
            }
        }
    }
    else
    {
        std::generate(data.begin(), data.end(), [&]() { return static_cast<T>(distribution(gen)); });

    }
    return data;
}

template<class T>
inline auto get_random_data(size_t size, T min, T max, seed_type seed_value)
    -> typename std::enable_if<
        is_custom_test_type<T>::value && std::is_integral<typename T::value_type>::value,
        std::vector<T>
        >::type
{
    engine_type gen(seed_value);
    std::uniform_int_distribution<typename T::value_type> distribution(min.x, max.x);
    std::vector<T> data(size);
    size_t segment_size = size / random_data_generation_segments;
    if(segment_size != 0)
    {
        for(uint32_t segment_index = 0; segment_index < random_data_generation_segments; segment_index++)
        {
            if(segment_index % random_data_generation_repeat_strides == 0)
            {
                T repeated_value = T(distribution(gen), distribution(gen));
                std::fill(
                    data.begin() + segment_size * segment_index,
                    data.begin() + segment_size * (segment_index + 1),
                    repeated_value);

            }
            else
            {
                std::generate(
                    data.begin() + segment_size * segment_index,
                    data.begin() + segment_size * (segment_index + 1),
                    [&]() { return T(distribution(gen), distribution(gen)); });
            }
        }
    }
    else
    {
        std::generate(data.begin(), data.end(), [&]() { return T(distribution(gen), distribution(gen)); });
    }
    return data;
}

template<class T>
inline auto get_random_data(size_t size, T min, T max, seed_type seed_value)
    -> typename std::enable_if<
        is_custom_test_type<T>::value && std::is_floating_point<typename T::value_type>::value,
        std::vector<T>
        >::type
{
    engine_type gen(seed_value);
    std::uniform_real_distribution<typename T::value_type> distribution(min.x, max.x);
    std::vector<T> data(size);
    size_t segment_size = size / random_data_generation_segments;
    if(segment_size != 0)
    {
        for(uint32_t segment_index = 0; segment_index < random_data_generation_segments; segment_index++)
        {
            if(segment_index % random_data_generation_repeat_strides == 0)
            {
                T repeated_value = T(distribution(gen), distribution(gen));
                std::fill(
                    data.begin() + segment_size * segment_index,
                    data.begin() + segment_size * (segment_index + 1),
                    repeated_value);

            }
            else
            {
                std::generate(
                    data.begin() + segment_size * segment_index,
                    data.begin() + segment_size * (segment_index + 1),
                    [&]() { return T(distribution(gen), distribution(gen)); });
            }
        }
    }
    else
    {
        std::generate(data.begin(), data.end(), [&]() { return T(distribution(gen), distribution(gen)); });
    }
    return data;
}

template<class T>
inline auto get_random_data(size_t size, typename T::value_type min, typename T::value_type max, seed_type seed_value)
    -> typename std::enable_if<
        is_custom_test_array_type<T>::value && std::is_integral<typename T::value_type>::value,
        std::vector<T>
        >::type
{
    engine_type gen(seed_value);
    std::uniform_int_distribution<typename T::value_type> distribution(min, max);
    std::vector<T> data(size);
    std::generate(
        data.begin(), data.end(),
        [&]()
        {
            T result;
            for(size_t i = 0; i < T::size; i++)
            {
                result.values[i] = distribution(gen);
            }
            return result;
        }
    );
    return data;
}

template<class T, class U, class V>
inline auto get_random_value(U min, V max, seed_type seed_value)
    -> typename std::enable_if<rocprim::is_arithmetic<T>::value, T>::type
{
    return get_random_data<T>(random_data_generation_segments, min, max, seed_value)[0];
}

template<class T>
inline auto get_random_value(typename T::value_type min, typename T::value_type max, seed_type seed_value)
    -> typename std::enable_if<is_custom_test_type<T>::value || is_custom_test_array_type<T>::value, T>::type
{
    return get_random_data<typename T::value_type>(random_data_generation_segments, min, max, seed_value)[0];
}

template<class T>
inline std::vector<T> get_random_data01(size_t size, float p, seed_type seed_value)
{
    const size_t max_random_size = 1024 * 1024;
    engine_type gen{seed_value};
    std::bernoulli_distribution distribution(p);
    std::vector<T> data(size);
    std::generate(
        data.begin(), data.begin() + std::min(size, max_random_size),
        [&]() { return static_cast<T>(distribution(gen)); }
    );
    for(size_t i = max_random_size; i < size; i += max_random_size)
    {
        std::copy_n(data.begin(), std::min(size - i, max_random_size), data.begin() + i);
    }
    return data;
}

}

#endif //ROCPRIM_TEST_UTILS_DATA_GENERATION_HPP

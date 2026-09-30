// Copyright (c) 2017-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include "primbench.hpp"

#include "../common/utils.hpp"
#include "../common/utils_custom_type.hpp"
#include "../common/utils_data_generation.hpp"
#include "../common/utils_half.hpp"

#include <rocprim/block/block_load.hpp>
#include <rocprim/block/block_scan.hpp>
#include <rocprim/config.hpp>
#include <rocprim/device/config_types.hpp>
#include <rocprim/device/detail/device_config_helper.hpp>
#include <rocprim/intrinsics/arch.hpp>
#include <rocprim/intrinsics/thread.hpp>
#include <rocprim/type_traits.hpp>
#include <rocprim/types.hpp>
#include <rocprim/types/tuple.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

/** \brief Maximum shared memory (in bytes) used for kernel tuning.
 *
 *  65536 bytes equals 64 KiB, which is the typical per-block shared memory limit on CUDA
 *  and the per-work-group LDS limit on HIP for most GPUs up to CDNA3.
 *
 *  \note Future GPU architectures do support more than 64 KiB of shared
 *        memory per block or work-group. This value should eventually be obtained
 *        dynamically from device properties such as hipDeviceProp_t::sharedMemPerBlock
 *        instead of being hardcoded.
 */
#ifndef TUNING_SHARED_MEMORY_MAX
    #define TUNING_SHARED_MEMORY_MAX 65536u
#endif

struct half_less
{
    ROCPRIM_HOST_DEVICE
    inline bool
        operator()(const rocprim::half& a, const rocprim::half& b) const
    {
#if __HIP_DEVICE_COMPILE__
        return a < b;
#else
        return common::half_to_native(a) < common::half_to_native(b);
#endif
    }
};

struct half_plus
{
    ROCPRIM_HOST_DEVICE
    inline rocprim::half
        operator()(const rocprim::half& a, const rocprim::half& b) const
    {
#if __HIP_DEVICE_COMPILE__
        return a + b;
#else
        return common::native_to_half(common::half_to_native(a) + common::half_to_native(b));
#endif
    }
};

struct half_equal_to
{
    ROCPRIM_HOST_DEVICE
    inline bool
        operator()(const rocprim::half& a, const rocprim::half& b) const
    {
#if __HIP_DEVICE_COMPILE__
        return a == b;
#else
        return common::half_to_native(a) == common::half_to_native(b);
#endif
    }
};

using engine_type = std::minstd_rand;

// generate_random_data_n() generates only part of sequence and replicates it,
// because benchmarks usually do not need "true" random sequence.
template<typename OutputIterator, typename U, typename V, typename Generator>
inline auto generate_random_data_n(OutputIterator it,
                                   size_t         size,
                                   U              min,
                                   V              max,
                                   Generator&     gen,
                                   size_t         max_random_size = 1024 * 1024)
    -> std::enable_if_t<rocprim::is_arithmetic<common::it_value_t<OutputIterator>>::value,
                        OutputIterator>
{
    using value_type = common::it_value_t<OutputIterator>;

    // Ensure value_type is valid for distribution
    using dis_type = std::conditional_t<
        rocprim::is_integral<value_type>::value,

        // Integral
        std::conditional_t<
            common::is_valid_for_int_distribution<value_type>::value,
            value_type,
            std::conditional_t<std::is_signed<value_type>::value, int, unsigned int>>,

        // Floating point
        std::conditional_t<std::is_same_v<rocprim::half, value_type>
                               || std::is_same_v<rocprim::bfloat16, value_type>,
                           float,
                           value_type>>;

    using distribution_type =
        typename std::conditional<rocprim::is_integral<dis_type>::value,
                                  common::uniform_int_distribution<dis_type>,
                                  std::uniform_real_distribution<dis_type>>::type;

    distribution_type distribution(min, max);

    std::generate_n(it, std::min(size, max_random_size), [&]() { return distribution(gen); });
    for(size_t i = max_random_size; i < size; i += max_random_size)
    {
        std::copy_n(it, std::min(size - i, max_random_size), it + i);
    }
    return it + size;
}

// Non arithmetic types
template<typename OutputIterator, typename Generator>
inline auto generate_random_data_n(OutputIterator                     it,
                                   size_t                             size,
                                   common::it_value_t<OutputIterator> min,
                                   common::it_value_t<OutputIterator> max,
                                   Generator&                         gen,
                                   size_t                             max_random_size = 1024 * 1024)
    -> std::enable_if_t<!rocprim::is_arithmetic<common::it_value_t<OutputIterator>>::value,
                        OutputIterator>
{
    using T = common::it_value_t<OutputIterator>;

    if constexpr(common::is_custom_type<T>::value)
    {
        using first_type  = typename T::first_type;
        using second_type = typename T::second_type;

        std::vector<first_type>  fdata(size);
        std::vector<second_type> sdata(size);
        generate_random_data_n(fdata.begin(), size, min.x, max.x, gen, max_random_size);
        generate_random_data_n(sdata.begin(), size, min.y, max.y, gen, max_random_size);

        for(size_t i = 0; i < size; ++i)
        {
            it[i] = T(fdata[i], sdata[i]);
        }
    }
    else
    {
        static_assert(!std::is_same<decltype(max.x), void>(), "Custom types must have an x field");

        using field_type = decltype(max.x);
        std::vector<field_type> field_data(size);
        generate_random_data_n(field_data.begin(), size, min.x, max.x, gen, max_random_size);

        for(size_t i = 0; i < size; ++i)
        {
            it[i] = T(field_data[i]);
        }
    }

    return it + size;
}

template<typename T>
inline std::vector<T>
    get_random_data01(size_t size, float p, unsigned int seed, size_t max_random_size = 1024 * 1024)
{
    engine_type                 gen(seed);
    std::bernoulli_distribution distribution(p);
    std::vector<T>              data(size);
    std::generate(data.begin(),
                  data.begin() + std::min(size, max_random_size),
                  [&]() { return distribution(gen); });
    for(size_t i = max_random_size; i < size; i += max_random_size)
    {
        std::copy_n(data.begin(), std::min(size - i, max_random_size), data.begin() + i);
    }
    return data;
}

template<typename T, typename U>
struct is_comparable
{
private:
    // A dummy template function that attempts to compare two objects of types T and U
    template<typename V, typename W>
    static auto test(V&& v, W&& w)
        -> decltype(std::declval<V>() < std::declval<W>(), std::true_type{});

    // Fallback if the above template function is not valid
    template<typename, typename>
    static std::false_type test(...);

public:
    // Final result
    static constexpr bool value = decltype(test<T, U>(std::declval<T>(), std::declval<U>()))::value;
};

template<typename T, typename U, typename V>
struct is_comparable<common::custom_type<U, V>, T>
    : std::conditional_t<rocprim::is_arithmetic<T>::value
                             || !std::is_same<T, common::custom_type<U, V>>::value,
                         std::false_type,
                         std::true_type>
{};

template<typename CustomType>
struct custom_type_decomposer
{
    static_assert(
        common::is_custom_type<CustomType>::value,
        "custom_type_decomposer can only be used with instantiations of common::custom_type");

    using T = typename CustomType::first_type;
    using U = typename CustomType::second_type;

    __host__ __device__
    ::rocprim::tuple<T&, U&>
        operator()(CustomType& key) const
    {
        return ::rocprim::tuple<T&, U&>{key.x, key.y};
    }
};

namespace common
{

template<typename T>
struct generate_limits<T, std::enable_if_t<common::is_custom_type<T>::value>>
{
    using F = typename T::first_type;
    using S = typename T::second_type;
    static inline T min()
    {
        return T(generate_limits<F>::min(), generate_limits<S>::min());
    }
    static inline T max()
    {
        return T(generate_limits<F>::max(), generate_limits<S>::max());
    }
};

} // namespace common

template<typename T, typename U, typename V>
inline std::vector<T> get_random_data(
    size_t size, U min, V max, unsigned int seed, size_t max_random_size = 1024 * 1024)
{
    std::vector<T> data(size);
    engine_type    gen(seed);
    generate_random_data_n(data.begin(), size, min, max, gen, max_random_size);
    return data;
}

template<typename T, typename U>
auto limit_cast(U value) -> T
{
    static_assert(rocprim::is_arithmetic<T>::value && rocprim::is_arithmetic<U>::value
                      && is_comparable<T, U>::value,
                  "Cannot use limit_cast with chosen types of T and U");

    using common_type = typename std::common_type<T, U>::type;
    if(rocprim::is_unsigned<T>::value)
    {
        if(value < 0)
        {
            return rocprim::numeric_limits<T>::min();
        }
        if(static_cast<common_type>(value)
           > static_cast<common_type>(rocprim::numeric_limits<T>::max()))
        {
            return rocprim::numeric_limits<T>::max();
        }
    }
    else if(rocprim::is_signed<T>::value && rocprim::is_unsigned<U>::value)
    {
        if(value > rocprim::numeric_limits<T>::max())
        {
            return rocprim::numeric_limits<T>::max();
        }
    }
    else if(rocprim::is_floating_point<T>::value)
    {
        return static_cast<T>(value);
    }
    else // Both T and U are signed
    {
        if(value < static_cast<common_type>(rocprim::numeric_limits<T>::min()))
        {
            return rocprim::numeric_limits<T>::min();
        }
        else if(value > static_cast<common_type>(rocprim::numeric_limits<T>::max()))
        {
            return rocprim::numeric_limits<T>::max();
        }
    }
    return static_cast<T>(value);
}

template<typename T, typename U, typename V>
inline auto limit_random_range(U range_start, V range_end)
{
    if constexpr(common::is_custom_type<T>::value)
    {
        static_assert(is_comparable<typename T::first_type, U>::value,
                      "T::first_type must be comparable with U");
        static_assert(is_comparable<typename T::second_type, U>::value,
                      "T::second_type must be comparable with U");
        static_assert(is_comparable<typename T::first_type, V>::value,
                      "T::first_type must be comparable with V");
        static_assert(is_comparable<typename T::second_type, V>::value,
                      "T::second_type must be comparable with V");
        static_assert(rocprim::is_arithmetic<typename T::first_type>::value,
                      "T::first_type must be arithmetic");
        static_assert(rocprim::is_arithmetic<typename T::second_type>::value,
                      "T::second_type must be arithmetic");
        static_assert(rocprim::is_arithmetic<U>::value, "U must be arithmetic");
        static_assert(rocprim::is_arithmetic<V>::value, "V must be arithmetic");

        return std::pair<T, T>{
            T{limit_cast<typename T::first_type>(range_start),
              limit_cast<typename T::second_type>(range_start)},
            T{  limit_cast<typename T::first_type>(range_end),
              limit_cast<typename T::second_type>(range_end)  }
        };
    }
    else if constexpr(is_comparable<T, U>::value && is_comparable<T, V>::value)
    {
        if constexpr(is_comparable<V, U>::value)
        {
            using common_type = typename std::common_type<T, U>::type;
            if(static_cast<common_type>(range_start) > static_cast<common_type>(range_end))
            {
                throw std::range_error("limit_random_range: Incorrect range used!");
            }
        }

        T start = limit_cast<T>(range_start);
        T end   = limit_cast<T>(range_end);
        return std::make_pair(start, end);
    }
    // Selected for non-standard float types, e.g. half, which cannot be compared with the limit types.
    return std::pair<T, T>{static_cast<T>(range_start), static_cast<T>(range_end)};
}

inline bool is_warp_size_supported(const unsigned int required_warp_size, const int device_id)
{
    unsigned int warp_size;
    HIP_CHECK(::rocprim::host_warp_size(device_id, warp_size));
    return warp_size >= required_warp_size;
}

/// \brief Get segments of uniform random size in [1, max_segment_length] with random key.
template<typename T>
std::vector<T>
    get_random_segments(const size_t size, const size_t max_segment_length, unsigned int seed)
{
    static_assert(rocprim::is_arithmetic<T>::value, "Key type must be arithmetic");

    engine_type                              prng(seed);
    common::uniform_int_distribution<size_t> segment_length_distribution(
        std::numeric_limits<size_t>::min(),
        max_segment_length);
    // std::uniform_real_distribution cannot handle:
    //   - rocprim::half, use float instead.
    //   - single byte integer-based types (char, unsigned char, int8_t, uint8_t), use int instead
    using dis_type = typename std::conditional<
        std::is_same<rocprim::half, T>::value,
        float,
        typename std::conditional<
            rocprim::is_integral<T>::value && !common::is_valid_for_int_distribution<T>::value,
            typename std::conditional<std::is_signed<T>::value, int, unsigned int>::type,
            T>::type>::type;

    using key_distribution_type = std::conditional_t<rocprim::is_integral<T>::value,
                                                     common::uniform_int_distribution<dis_type>,
                                                     std::uniform_real_distribution<dis_type>>;
    key_distribution_type key_distribution(rocprim::numeric_limits<T>::max());
    std::vector<T>        keys(size);

    size_t keys_start_index = 0;
    while(keys_start_index < size)
    {
        const size_t new_segment_length = segment_length_distribution(prng);
        const size_t new_segment_end    = std::min(size, keys_start_index + new_segment_length);
        const T      key                = key_distribution(prng);
        std::fill(keys.begin() + keys_start_index, keys.begin() + new_segment_end, key);
        keys_start_index += new_segment_length;
    }
    return keys;
}

/// \brief Get segments of uniform random size in [1, max_segment_length] with unique incrementing key.
template<typename T>
std::vector<T>
    get_random_segments_iota(const size_t size, const size_t max_segment_length, unsigned int seed)
{
    engine_type                              prng(seed);
    common::uniform_int_distribution<size_t> segment_length_distribution(1, max_segment_length);

    std::vector<T> keys(size);

    size_t segment_index    = 0;
    size_t keys_start_index = 0;
    while(keys_start_index < size)
    {
        const size_t new_segment_length = segment_length_distribution(prng);
        const size_t new_segment_end    = std::min(size, keys_start_index + new_segment_length);
        const T      key                = segment_index++;
        std::fill(keys.begin() + keys_start_index, keys.begin() + new_segment_end, key);
        keys_start_index += new_segment_length;
    }
    return keys;
}

template<typename T, typename U = rocprim::empty_type, typename V = rocprim::empty_type>
inline auto get_random_value(U min, V max, size_t seed_value)
{
    if constexpr(rocprim::is_arithmetic<T>::value)
    {
        T           result;
        engine_type gen(seed_value);
        generate_random_data_n(&result, 1, min, max, gen);
        return result;
    }
    if constexpr(common::is_custom_type<T>::value)
    {
        typename T::first_type  result_first;
        typename T::second_type result_second;
        engine_type             gen(seed_value);
        generate_random_data_n(&result_first, 1, min.x, max.x, gen);
        generate_random_data_n(&result_second, 1, min.y, max.y, gen);
        return T{result_first, result_second};
    }
}

template<typename T, T, typename>
struct make_index_range_impl;

template<typename T, T Start, T... I>
struct make_index_range_impl<T, Start, std::integer_sequence<T, I...>>
{
    using type = std::integer_sequence<T, (Start + I)...>;
};

// make a std::integer_sequence with values from Start to End inclusive
template<typename T, T Start, T End>
using make_index_range =
    typename make_index_range_impl<T, Start, std::make_integer_sequence<T, End - Start + 1>>::type;

template<typename T, template<T> class Function, T... I, typename... Args>
void static_for_each_impl(std::integer_sequence<T, I...>, Args&&... args)
{
    int a[] = {(Function<I>{}(std::forward<Args>(args)...), 0)...};
    static_cast<void>(a);
}

// call the supplied template with all values of the std::integer_sequence Indices
template<typename Indices, template<typename Indices::value_type> class Function, typename... Args>
void static_for_each(Args&&... args)
{
    static_for_each_impl<typename Indices::value_type, Function>(Indices{},
                                                                 std::forward<Args>(args)...);
}

// Inserts spaces at beginning of string if string shorter than specified length.
inline std::string pad_string(std::string str, const size_t len)
{
    if(len > str.size())
    {
        str.insert(str.begin(), len - str.size(), ' ');
    }

    return str;
}

template<std::size_t Size, std::size_t Alignment>
struct alignas(Alignment) custom_aligned_type
{
    unsigned char data[Size];
};

PRIMBENCH_REGISTER_TYPE(int8_t, "i8")
PRIMBENCH_REGISTER_TYPE(int16_t, "i16")
PRIMBENCH_REGISTER_TYPE(int32_t, "i32")
PRIMBENCH_REGISTER_TYPE(int64_t, "i64")

PRIMBENCH_REGISTER_TYPE(uint8_t, "u8")
PRIMBENCH_REGISTER_TYPE(uint32_t, "u32")
PRIMBENCH_REGISTER_TYPE(uint64_t, "u64")

PRIMBENCH_REGISTER_TYPE(float, "f32")
PRIMBENCH_REGISTER_TYPE(double, "f64")

PRIMBENCH_REGISTER_TYPE(float2, "float2")
PRIMBENCH_REGISTER_TYPE(float4, "float4")
PRIMBENCH_REGISTER_TYPE(double2, "double2")

PRIMBENCH_REGISTER_TYPE(rocprim::half, "half")
PRIMBENCH_REGISTER_TYPE(rocprim::empty_type, "empty")
PRIMBENCH_REGISTER_TYPE(rocprim::int128_t, "i128")
PRIMBENCH_REGISTER_TYPE(rocprim::uint128_t, "u128")

using custom_i32_i32 = common::custom_type<int32_t, int32_t>;
PRIMBENCH_REGISTER_TYPE(custom_i32_i32, "custom<i32,i32>")

using custom_f32_f32 = common::custom_type<float, float>;
PRIMBENCH_REGISTER_TYPE(custom_f32_f32, "custom<f32,f32>")

using custom_f64_f64 = common::custom_type<double, double>;
PRIMBENCH_REGISTER_TYPE(custom_f64_f64, "custom<f64,f64>")

using custom_i32_f64 = common::custom_type<int32_t, double>;
PRIMBENCH_REGISTER_TYPE(custom_i32_f64, "custom<i32,f64>")

using custom_i8_f64 = common::custom_type<int8_t, double>;
PRIMBENCH_REGISTER_TYPE(custom_i8_f64, "custom<i8,f64>")

using custom_i8_i16 = common::custom_type<int8_t, int16_t>;
PRIMBENCH_REGISTER_TYPE(custom_i8_i16, "custom<i8,i16>")

using custom_i64_f64 = common::custom_type<int64_t, double>;
PRIMBENCH_REGISTER_TYPE(custom_i64_f64, "custom<i64,f64>")

using custom_f32_i16 = common::custom_type<float, int16_t>;
PRIMBENCH_REGISTER_TYPE(custom_f32_i16, "custom<f32,i16>")

using custom_i64_i64 = common::custom_type<int64_t, int64_t>;
PRIMBENCH_REGISTER_TYPE(custom_i64_i64, "custom<i64,i64>")

using copyable_i8_f64 = common::custom_type_copyable<int8_t, double>;
PRIMBENCH_REGISTER_TYPE(copyable_i8_f64, "copyable<i8,f64>")

using copyable_f64_f64 = common::custom_type_copyable<double, double>;
PRIMBENCH_REGISTER_TYPE(copyable_f64_f64, "copyable<f64,f64>")

using huge_1024_f32_f32 = common::custom_huge_type<1024, float, float>;
PRIMBENCH_REGISTER_TYPE(huge_1024_f32_f32, "huge<1024,f32,f32>")

using huge_2048_f32_f32 = common::custom_huge_type<2048, float, float>;
PRIMBENCH_REGISTER_TYPE(huge_2048_f32_f32, "huge<2048,f32,f32>")

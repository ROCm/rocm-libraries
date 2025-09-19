// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BENCHMARK_UTILS_HPP_
#define ROCPRIM_BENCHMARK_UTILS_HPP_

#include "../common/utils.hpp"
#include "../common/utils_custom_type.hpp"
#include "../common/utils_data_generation.hpp"
#include "../common/utils_half.hpp"

#include <benchmark/benchmark.h>

// rocPRIM
#include <rocprim/block/block_load.hpp>
#include <rocprim/block/block_scan.hpp>
#include <rocprim/config.hpp>
#include <rocprim/device/config_types.hpp>
#include <rocprim/device/detail/device_config_helper.hpp> // partition_config_params
#include <rocprim/intrinsics/arch.hpp>
#include <rocprim/intrinsics/thread.hpp>
#include <rocprim/type_traits.hpp>
#include <rocprim/types.hpp>
#include <rocprim/types/tuple.hpp>

// CmdParser
#include "cmdparser.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
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

#define TUNING_SHARED_MEMORY_MAX 65536u
// Support half operators on host side

inline const char* get_seed_message()
{
    return "seed for input generation, either an unsigned integer value for determinisic results "
           "or 'random' for different inputs for each repetition";
}

/// \brief Provides a sequence of seeds.
class managed_seed
{
public:
    /// \param[in] seed_string Either "random" to get random seeds,
    ///   or an unsigned integer to get (a sequence) of deterministic seeds.
    managed_seed(const std::string& seed_string)
    {
        is_random = seed_string == "random";
        if(!is_random)
        {
            const unsigned int seed = std::stoul(seed_string);
            std::seed_seq      seq{seed};
            seq.generate(seeds.begin(), seeds.end());
        }
    }

    managed_seed() {}

    unsigned int get_0() const
    {
        return is_random ? std::random_device{}() : seeds[0];
    }

    unsigned int get_1() const
    {
        return is_random ? std::random_device{}() : seeds[1];
    }

    unsigned int get_2() const
    {
        return is_random ? std::random_device{}() : seeds[2];
    }

private:
    std::array<unsigned int, 3> seeds;
    bool                        is_random;
};

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
template<typename OutputIter, typename U, typename V, typename Generator>
inline auto generate_random_data_n(
    OutputIter it, size_t size, U min, V max, Generator& gen, size_t max_random_size = 1024 * 1024)
    -> typename std::enable_if_t<rocprim::is_integral<common::it_value_t<OutputIter>>::value,
                                 OutputIter>
{
    using T = common::it_value_t<OutputIter>;

    using dis_type = typename std::conditional<
        common::is_valid_for_int_distribution<T>::value,
        T,
        typename std::conditional<std::is_signed<T>::value, int, unsigned int>::type>::type;
    common::uniform_int_distribution<dis_type> distribution((T)min, (T)max);
    std::generate_n(it, std::min(size, max_random_size), [&]() { return distribution(gen); });
    for(size_t i = max_random_size; i < size; i += max_random_size)
    {
        std::copy_n(it, std::min(size - i, max_random_size), it + i);
    }
    return it + size;
}

template<typename OutputIterator, typename U, typename V, typename Generator>
inline auto generate_random_data_n(OutputIterator it,
                                   size_t         size,
                                   U              min,
                                   V              max,
                                   Generator&     gen,
                                   size_t         max_random_size = 1024 * 1024)
    -> std::enable_if_t<rocprim::is_floating_point<common::it_value_t<OutputIterator>>::value,
                        OutputIterator>
{
    using T = typename std::iterator_traits<OutputIterator>::value_type;

    // Generate floats when T is half
    using dis_type = std::conditional_t<std::is_same<rocprim::half, T>::value
                                            || std::is_same<rocprim::bfloat16, T>::value,
                                        float,
                                        T>;
    std::uniform_real_distribution<dis_type> distribution((dis_type)min, (dis_type)max);
    std::generate_n(it, std::min(size, max_random_size), [&]() { return distribution(gen); });
    for(size_t i = max_random_size; i < size; i += max_random_size)
    {
        std::copy_n(it, std::min(size - i, max_random_size), it + i);
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

template<typename OutputIterator, typename Generator>
inline auto generate_random_data_n(OutputIterator                     it,
                                   size_t                             size,
                                   common::it_value_t<OutputIterator> min,
                                   common::it_value_t<OutputIterator> max,
                                   Generator&                         gen,
                                   size_t                             max_random_size = 1024 * 1024)
    -> std::enable_if_t<common::is_custom_type<common::it_value_t<OutputIterator>>::value,
                        OutputIterator>
{
    using T = common::it_value_t<OutputIterator>;

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
    return it + size;
}

template<typename OutputIterator, typename Generator>
inline auto generate_random_data_n(OutputIterator                     it,
                                   size_t                             size,
                                   common::it_value_t<OutputIterator> min,
                                   common::it_value_t<OutputIterator> max,
                                   Generator&                         gen,
                                   size_t                             max_random_size = 1024 * 1024)
    -> std::enable_if_t<!common::is_custom_type<common::it_value_t<OutputIterator>>::value
                            && !std::is_same<decltype(max.x), void>::value,
                        OutputIterator>
{
    using T = common::it_value_t<OutputIterator>;

    using field_type = decltype(max.x);
    std::vector<field_type> field_data(size);
    generate_random_data_n(field_data.begin(), size, min.x, max.x, gen, max_random_size);
    for(size_t i = 0; i < size; ++i)
    {
        it[i] = T(field_data[i]);
    }
    return it + size;
}

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

// This overload below is selected for non-standard float types, e.g. half, which cannot be compared with the limit types.
template<typename T, typename U, typename V>
inline auto limit_random_range(U range_start, V range_end)
    -> std::enable_if_t<!common::is_custom_type<T>::value
                            && (!is_comparable<T, U>::value || !is_comparable<T, V>::value),
                        std::pair<T, T>>
{
    return {static_cast<T>(range_start), static_cast<T>(range_end)};
}

template<typename T, typename U, typename V>
auto limit_random_range(U range_start, V range_end)
    -> std::enable_if_t<(common::is_custom_type<T>::value
                         && is_comparable<typename T::first_type, U>::value
                         && is_comparable<typename T::second_type, U>::value
                         && is_comparable<typename T::first_type, V>::value
                         && is_comparable<typename T::second_type, V>::value
                         && rocprim::is_arithmetic<typename T::first_type>::value
                         && rocprim::is_arithmetic<typename T::second_type>::value
                         && rocprim::is_arithmetic<U>::value && rocprim::is_arithmetic<V>::value),
                        std::pair<T, T>>
{

    return {
        T{limit_cast<typename T::first_type>(range_start),
          limit_cast<typename T::second_type>(range_start)},
        T{  limit_cast<typename T::first_type>(range_end),
          limit_cast<typename T::second_type>(range_end)  }
    };
}

template<typename T, typename U, typename V>
inline auto limit_random_range(U range_start, V range_end)
    -> std::enable_if_t<!common::is_custom_type<T>::value && is_comparable<T, U>::value
                            && is_comparable<T, V>::value,
                        std::pair<T, T>>
{

    if(is_comparable<V, U>::value)
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
    // std::uniform_real_distribution cannot handle rocprim::half, use float instead
    using dis_type =
        typename std::conditional<std::is_same<rocprim::half, T>::value, float, T>::type;
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

template<typename T, typename U, typename V>
inline auto get_random_value(U min, V max, size_t seed_value)
    -> std::enable_if_t<rocprim::is_arithmetic<T>::value, T>
{
    T           result;
    engine_type gen(seed_value);
    generate_random_data_n(&result, 1, min, max, gen);
    return result;
}

template<typename T>
inline auto get_random_value(T min, T max, size_t seed_value)
    -> std::enable_if_t<common::is_custom_type<T>::value, T>
{
    typename T::first_type  result_first;
    typename T::second_type result_second;
    engine_type             gen(seed_value);
    generate_random_data_n(&result_first, 1, min.x, max.x, gen);
    generate_random_data_n(&result_second, 1, min.y, max.y, gen);
    return T{result_first, result_second};
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

struct bench_naming
{
public:
    enum format
    {
        json,
        human,
        txt
    };
    static format& get_format()
    {
        static format storage = human;
        return storage;
    }
    static void set_format(const std::string& argument)
    {
        format result = human;
        if(argument == "json")
        {
            result = json;
        }
        else if(argument == "txt")
        {
            result = txt;
        }
        get_format() = result;
    }

private:
    static std::string matches_as_json(std::sregex_iterator& matches)
    {
        std::stringstream result;
        int               brackets_count = 1;
        result << "{";
        bool insert_comma = false;
        for(std::sregex_iterator i = matches; i != std::sregex_iterator(); ++i)
        {
            std::smatch m = *i;
            if(insert_comma)
            {
                result << ",";
            }
            else
            {
                insert_comma = true;
            }
            result << "\"" << m[1].str() << "\":";
            if(m[2].length() > 0)
            {
                if(m[2].str().find_first_not_of("0123456789") == std::string::npos)
                {
                    result << m[2].str();
                }
                else
                {
                    result << "\"" << m[2].str() << "\"";
                }
                if(m[3].length() > 0 && brackets_count > 0)
                {
                    int n = std::min(brackets_count, static_cast<int>(m[3].length()));
                    brackets_count -= n;
                    for(int c = 0; c < n; ++c)
                    {
                        result << "}";
                    }
                }
            }
            else
            {
                ++brackets_count;
                result << "{";
                insert_comma = false;
            }
        }
        while(brackets_count > 0)
        {
            --brackets_count;
            result << "}";
        }
        return result.str();
    }

    static std::string matches_as_human(std::sregex_iterator& matches)
    {
        std::stringstream result;
        int               brackets_count = 0;
        bool              insert_comma   = false;
        for(std::sregex_iterator i = matches; i != std::sregex_iterator(); ++i)
        {
            std::smatch m = *i;
            if(insert_comma)
            {
                result << ",";
            }
            else
            {
                insert_comma = true;
            }
            if(m[2].length() > 0)
            {
                result << m[2].str();
                if(m[3].length() > 0 && brackets_count > 0)
                {
                    int n = std::min(brackets_count, static_cast<int>(m[3].length()));
                    brackets_count -= n;
                    for(int c = 0; c < n; ++c)
                    {
                        result << ">";
                    }
                }
            }
            else
            {
                ++brackets_count;
                result << "<";
                insert_comma = false;
            }
        }
        while(brackets_count > 0)
        {
            --brackets_count;
            result << ">";
        }
        return result.str();
    }

public:
    static std::string format_name(std::string string)
    {
        format     format = get_format();
        std::regex r("([A-z0-9]*):\\s*((?:common::custom_type<[A-z0-9,]*>)|(?:common::custom_type_"
                     "copyable<[A-z0-9,]*>)|[A-z:\\(\\)\\.<>\\s0-9]*)(\\}*)");
        // First we perform some checks
        bool checks[4] = {false};
        for(std::sregex_iterator i = std::sregex_iterator(string.begin(), string.end(), r);
            i != std::sregex_iterator();
            ++i)
        {
            std::smatch m = *i;
            if(m[1].str() == "lvl")
            {
                checks[0] = true;
            }
            else if(m[1].str() == "algo")
            {
                checks[1] = true;
            }
            else if(m[1].str() == "cfg")
            {
                checks[2] = true;
            }
        }
        std::string string_substitute = std::regex_replace(string, r, "");
        checks[3] = string_substitute.find_first_not_of(" ,{}") == std::string::npos;
        for(bool check_name_format : checks)
        {
            if(!check_name_format)
            {
                std::cout << "Benchmark name \"" << string
                          << "\" not in the correct format (e.g. "
                             "{lvl:block,algo:reduce,cfg:default_config} )"
                          << std::endl;
                exit(1);
            }
        }

        // Now we generate the desired format
        std::sregex_iterator matches = std::sregex_iterator(string.begin(), string.end(), r);

        switch(format)
        {
            case format::json: return matches_as_json(matches);
            case format::human: return matches_as_human(matches);
            case format::txt: return string;
        }
        return string;
    }
};

template<typename T>
struct Traits
{
    //static inline method instead of static inline attribute because that's only supported from C++17 onwards
    static inline const char* name()
    {
        static_assert(sizeof(T) == 0, "Traits<T>::name() unknown");
        return "unknown";
    }
};

// Explicit definitions
template<>
inline const char* Traits<char>::name()
{
    return "char";
}
template<>
inline const char* Traits<int>::name()
{
    return "int";
}
template<>
inline const char* Traits<short>::name()
{
    return "short";
}
template<>
inline const char* Traits<int8_t>::name()
{
    return "int8_t";
}
template<>
inline const char* Traits<uint8_t>::name()
{
    return "uint8_t";
}
template<>
inline const char* Traits<uint16_t>::name()
{
    return "uint16_t";
}
template<>
inline const char* Traits<uint32_t>::name()
{
    return "uint32_t";
}
template<>
inline const char* Traits<rocprim::half>::name()
{
    return "rocprim::half";
}
template<>
inline const char* Traits<rocprim::bfloat16>::name()
{
    return "rocprim::bfloat16";
}
template<>
inline const char* Traits<long long>::name()
{
    return "int64_t";
}
// On MSVC `int64_t` and `long long` are the same, leading to multiple definition errors
#ifndef _WIN32
template<>
inline const char* Traits<int64_t>::name()
{
    return "int64_t";
}
#endif
// On MSVC `uint64_t` and `unsigned long long` are the same, leading to multiple definition errors
#ifndef _WIN32
template<>
inline const char* Traits<uint64_t>::name()
{
    return "uint64_t";
}
#else
template<>
inline const char* Traits<unsigned long long>::name()
{
    return "unsigned long long";
}
#endif
template<>
inline const char* Traits<float>::name()
{
    return "float";
}
template<>
inline const char* Traits<double>::name()
{
    return "double";
}
template<>
inline const char* Traits<common::custom_type<int, int>>::name()
{
    return "common::custom_type<int,int>";
}
template<>
inline const char* Traits<common::custom_type<float, float>>::name()
{
    return "common::custom_type<float,float>";
}
template<>
inline const char* Traits<common::custom_huge_type<1024, float, float>>::name()
{
    return "common::custom_type<1024,float,float>";
}
template<>
inline const char* Traits<common::custom_huge_type<2048, float, float>>::name()
{
    return "common::custom_type<2048,float,float>";
}
template<>
inline const char* Traits<common::custom_type<double, double>>::name()
{
    return "common::custom_type<double,double>";
}
template<>
inline const char* Traits<common::custom_type<int, double>>::name()
{
    return "common::custom_type<int,double>";
}
template<>
inline const char* Traits<common::custom_type<char, double>>::name()
{
    return "common::custom_type<char,double>";
}
template<>
inline const char* Traits<common::custom_type<char, short>>::name()
{
    return "common::custom_type<char,short>";
}
template<>
inline const char* Traits<common::custom_type<long, double>>::name()
{
    return "common::custom_type<long,double>";
}
template<>
inline const char* Traits<common::custom_type<long long, double>>::name()
{
    return "common::custom_type<int64_t,double>";
}
template<>
inline const char* Traits<common::custom_type<float, int16_t>>::name()
{
    return "common::custom_type<float,int16_t>";
}
template<>
inline const char* Traits<rocprim::empty_type>::name()
{
    return "empty_type";
}
template<>
inline const char* Traits<HIP_vector_type<float, 2>>::name()
{
    return "float2";
}
template<>
inline const char* Traits<HIP_vector_type<double, 2>>::name()
{
    return "double2";
}
template<>
inline const char* Traits<rocprim::int128_t>::name()
{
    return "rocprim::int128_t";
}
template<>
inline const char* Traits<rocprim::uint128_t>::name()
{
    return "rocprim::uint128_t";
}
template<>
inline const char* Traits<common::custom_type_copyable<char, double>>::name()
{
    return "common::custom_type_copyable<char,double>";
}
template<>
inline const char* Traits<common::custom_type_copyable<double, double>>::name()
{
    return "common::custom_type_copyable<double,double>";
}

inline const char* get_block_scan_algorithm_name(rocprim::block_scan_algorithm alg)
{
    switch(alg)
    {
        case rocprim::block_scan_algorithm::using_warp_scan:
            return "block_scan_algorithm::using_warp_scan";
        case rocprim::block_scan_algorithm::reduce_then_scan:
            return "block_scan_algorithm::reduce_then_scan";
            // Not using `default: ...` because it kills effectiveness of -Wswitch
    }
    return "default_algorithm";
}

inline const char* get_block_load_method_name(rocprim::block_load_method method)
{
    switch(method)
    {
        case rocprim::block_load_method::block_load_direct:
            return "block_load_method::block_load_direct";
        case rocprim::block_load_method::block_load_striped:
            return "block_load_method::block_load_striped";
        case rocprim::block_load_method::block_load_vectorize:
            return "block_load_method::block_load_vectorize";
        case rocprim::block_load_method::block_load_transpose:
            return "block_load_method::block_load_transpose";
        case rocprim::block_load_method::block_load_warp_transpose:
            return "block_load_method::block_load_warp_transpose";
    }
    return "default_method";
}

inline const char* get_thread_load_method_name(rocprim::cache_load_modifier method)
{
    switch(method)
    {
        case rocprim::load_default: return "load_default";
        case rocprim::load_ca: return "load_ca";
        case rocprim::load_cg: return "load_cg";
        case rocprim::load_nontemporal: return "load_nontemporal";
        case rocprim::load_cv: return "load_cv";
        case rocprim::load_ldg: return "load_ldg";
        case rocprim::load_volatile: return "load_volatile";
        case rocprim::load_count: return "load_count";
    }
    return "load_default";
}

template<std::size_t Size, std::size_t Alignment>
struct alignas(Alignment) custom_aligned_type
{
    unsigned char data[Size];
};

template<typename Config>
std::string partition_config_name()
{
    const rocprim::detail::partition_config_params config = Config();
    return "{bs:" + std::to_string(config.kernel_config.block_size)
           + ",ipt:" + std::to_string(config.kernel_config.items_per_thread) + "}";
}

template<>
inline std::string partition_config_name<rocprim::default_config>()
{
    return "default_config";
}

namespace benchmark_utils
{

constexpr size_t KiB = 1024;
constexpr size_t MiB = 1024 * KiB;
constexpr size_t GiB = 1024 * MiB;

class state
{
public:
    state(hipStream_t         stream,
          size_t              size,
          const managed_seed& seed,
          size_t              batch_iterations,
          benchmark::State&   gbench_state,
          size_t              warmup_iterations,
          bool                cold,
          bool                record_as_whole)
        : stream(stream)
        , size(size)
        , bytes(size)
        , seed(seed)
        , batch_iterations(batch_iterations)
        , gbench_state(gbench_state)
        , warmup_iterations(warmup_iterations)
        , cold(cold)
        , record_as_whole(record_as_whole)
        , events(record_as_whole ? 2 : batch_iterations * 2)
    {}

    // Used to reset the input array of algorithms like device_merge_inplace.
    void run_before_every_iteration(std::function<void()> lambda)
    {
        run_before_every_iteration_lambda = lambda;
    }

    // Used to accumulate the results of state.run() calls.
    void accumulate_total_gbench_iterations_every_run()
    {
        reset_total_gbench_iterations_every_run = false;
    }

    void run(std::function<void()> kernel)
    {
        for(auto& event : events)
        {
            HIP_CHECK(hipEventCreate(&event));
        }

        // Warm-up
        for(size_t i = 0; i < warmup_iterations; ++i)
        {
            // Benchmarks may expect their kernel input to be prepared by this lambda,
            // so to prevent any potential crashes, we call the lambda during warm-up.
            if(run_before_every_iteration_lambda)
            {
                run_before_every_iteration_lambda();
            }

            kernel();
        }
        HIP_CHECK(hipDeviceSynchronize());

        if(run_before_every_iteration_lambda && batch_iterations > 1 && record_as_whole)
        {
            std::cerr << "Error: This benchmark calls run_before_every_iteration() and has a "
                         "batch_iterations count that is higher than 1, which means it does not "
                         "support using --record_as_whole.\n";
            exit(EXIT_FAILURE);
        }

        // Run
        for(auto _ : gbench_state)
        {
            if(record_as_whole)
            {
                if(run_before_every_iteration_lambda)
                {
                    run_before_every_iteration_lambda();
                }

                HIP_CHECK(hipEventRecord(events[0], stream));
                for(size_t i = 0; i < batch_iterations; ++i)
                {
                    kernel();
                }
                HIP_CHECK(hipEventRecord(events[1], stream));
                HIP_CHECK(hipEventSynchronize(events[1]));

                float elapsed_mseconds;
                HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, events[0], events[1]));
                times.emplace_back(elapsed_mseconds);
                gbench_state.SetIterationTime(elapsed_mseconds / 1000);
            }
            else
            {
                for(size_t i = 0; i < batch_iterations; ++i)
                {
                    if(run_before_every_iteration_lambda)
                    {
                        run_before_every_iteration_lambda();
                    }

                    if(cold)
                    {
                        clear_gpu_cache(stream);
                    }

                    // Even events record the start time.
                    HIP_CHECK(hipEventRecord(events[i * 2], stream));

                    kernel();

                    // Odd events record the stop time.
                    HIP_CHECK(hipEventRecord(events[i * 2 + 1], stream));
                }

                // Wait until the last record event has completed.
                HIP_CHECK(hipEventSynchronize(events[batch_iterations * 2 - 1]));

                // Accumulate the total elapsed time.
                double elapsed_mseconds = 0.0;
                for(size_t i = 0; i < batch_iterations; i++)
                {
                    float iteration_mseconds;
                    HIP_CHECK(
                        hipEventElapsedTime(&iteration_mseconds, events[i * 2], events[i * 2 + 1]));
                    times.emplace_back(iteration_mseconds);
                    elapsed_mseconds += iteration_mseconds;
                }
                gbench_state.SetIterationTime(elapsed_mseconds / 1000);
            }
        }

        if(reset_total_gbench_iterations_every_run)
        {
            total_gbench_iterations = 0;
        }
        total_gbench_iterations += gbench_state.iterations();

        for(const auto& event : events)
        {
            HIP_CHECK(hipEventDestroy(event));
        }
    }

    void set_throughput(size_t actual_size, size_t type_size)
    {
        if(has_set_throughput)
        {
            std::cerr << "Error: Benchmarks should only ever call set_throughput() once, at the "
                         "very end.\n";
            exit(EXIT_FAILURE);
        }
        has_set_throughput = true;

        gbench_state.SetBytesProcessed(total_gbench_iterations * batch_iterations * actual_size
                                       * type_size);
        gbench_state.SetItemsProcessed(total_gbench_iterations * batch_iterations * actual_size);

        output_statistics();
    }

    hipStream_t       stream;
    size_t            size;
    size_t            bytes;
    managed_seed      seed;
    size_t            batch_iterations;
    benchmark::State& gbench_state;

private:
    // Zeros a 256 MiB buffer, used to clear the cache before each kernel call.
    // 256 MiB is the size of the largest cache on any AMD GPU.
    // It is currently not possible to fetch the L3 cache size from the runtime.
    inline void clear_gpu_cache(hipStream_t stream)
    {
        constexpr size_t buf_size = 256 * MiB;
        static void*     buf      = nullptr;
        if(!buf)
        {
            HIP_CHECK(hipMalloc(&buf, buf_size));
        }
        HIP_CHECK(hipMemsetAsync(buf, 0, buf_size, stream));
    }

    void output_statistics()
    {
        double mean   = get_mean();
        double median = get_median();
        double stddev = get_stddev(mean);
        double cv     = get_cv(stddev, mean);

        gbench_state.counters["mean"]   = mean;
        gbench_state.counters["median"] = median;
        gbench_state.counters["stddev"] = stddev;
        gbench_state.counters["cv"]     = cv;
    }

    double get_mean()
    {
        return std::reduce(times.begin(), times.end()) / times.size();
    }

    // Technically when times.size() is even, the median is the arithmetic mean
    // of the elements k=N/2 and k=N/2+1. This would be overkill here,
    // as times.size() is large enough, and recorded times are similar enough.
    double get_median()
    {
        size_t center_index = times.size() / 2;
        std::nth_element(times.begin(), times.begin() + center_index, times.end());
        return times[center_index];
    }

    double get_stddev(double mean)
    {
        auto SumSquares = [](const std::vector<double>& v)
        { return std::transform_reduce(v.begin(), v.end(), v.begin(), 0.0); };
        auto Sqr  = [](double dat) { return dat * dat; };
        auto Sqrt = [](double dat) { return dat < 0.0 ? 0.0 : std::sqrt(dat); };

        double stddev = 0.0;
        if(times.size() > 1)
        {
            double avg_squares = SumSquares(times) * (1.0 / times.size());
            stddev = Sqrt(times.size() / (times.size() - 1.0) * (avg_squares - Sqr(mean)));
        }
        return stddev;
    }

    double get_cv(double stddev, double mean)
    {
        return times.size() >= 2 ? stddev / mean : 0.0;
    }

    size_t warmup_iterations;
    bool   cold;
    bool   record_as_whole;

    std::vector<hipEvent_t> events;
    std::function<void()>   run_before_every_iteration_lambda       = nullptr;
    size_t                  total_gbench_iterations                 = 0;
    bool                    reset_total_gbench_iterations_every_run = true;
    std::vector<double>     times;
    bool                    has_set_throughput = false;
};

struct autotune_interface
{
    virtual std::string name() const = 0;
    virtual std::string sort_key() const
    {
        return name();
    };
    virtual ~autotune_interface()   = default;
    virtual void run(state&& state) = 0;
};

class executor
{
public:
    executor(int    argc,
             char*  argv[],
             size_t default_bytes,
             size_t default_batch_iterations,
             size_t default_warmup_iterations,
             bool   default_cold   = true,
             int    default_trials = -1)
    {
        cli::Parser parser(argc, argv);

        set_optional_parser_flags(parser,
                                  default_bytes,
                                  default_batch_iterations,
                                  default_warmup_iterations,
                                  default_cold,
                                  default_trials);

        parser.run_and_exit_if_error();

        benchmark::Initialize(&argc, argv);

        parse(parser);

        add_context();
    }

    template<typename T>
    void queue_fn(const std::string& name, T bench_fn)
    {
        apply_settings(benchmark::RegisterBenchmark(name.c_str(),
                                                    [=](benchmark::State& gbench_state)
                                                    { bench_fn(new_state(gbench_state)); }));
    }

    template<typename Benchmark>
    void queue_instance(Benchmark&& instance)
    {
        apply_settings(benchmark::RegisterBenchmark(
            instance.name().c_str(),
            [=](benchmark::State& gbench_state)
            {
                // run() requires a mutable instance, so create a mutable copy.
                // Using [&instance] doesn't work, as it creates a dangling reference at runtime.
                // Marking the lambda mutable doesn't work, as the &&instance it copies is const.
                Benchmark(std::move(instance)).run(new_state(gbench_state));
            }));
    }

    template<typename Benchmark>
    static bool queue_sorted_instance()
    {
        sorted_benchmarks().push_back(std::make_unique<Benchmark>());
        return true; // Must return something, as this function gets called in global scope.
    }

    template<typename BulkCreateFunction>
    static bool queue_autotune(BulkCreateFunction&& f)
    {
        std::forward<BulkCreateFunction>(f)(sorted_benchmarks());
        return true; // Must return something, as this function gets called in global scope.
    }

    void run()
    {
        register_sorted_subset(parallel_instance, parallel_instances);
        benchmark::ConsoleReporter cr;
        benchmark::RunSpecifiedBenchmarks(&cr);
    }

private:
    void set_optional_parser_flags(cli::Parser& parser,
                                   size_t       default_bytes,
                                   size_t       default_batch_iterations,
                                   size_t       default_warmup_iterations,
                                   bool         default_cold,
                                   int          default_trials)
    {
        parser.set_optional<size_t>("size", "size", default_bytes, "size in bytes");
        parser.set_optional<size_t>("batch_iterations",
                                    "batch_iterations",
                                    default_batch_iterations,
                                    "number of batch iterations");
        parser.set_optional<size_t>("warmup_iterations",
                                    "warmup_iterations",
                                    default_warmup_iterations,
                                    "number of warmup iterations");
        parser.set_optional<bool>("hot",
                                  "hot",
                                  !default_cold,
                                  "don't clear the gpu cache on every batch iteration");
        parser.set_optional<bool>(
            "record_as_whole",
            "record_as_whole",
            false,
            "record the batch iterations as a whole, at the very start and end, which necessitates "
            "that gpu cache clearing between iterations can't be done");

        parser.set_optional<std::string>("seed", "seed", "random", get_seed_message());
        parser.set_optional<int>("trials", "trials", default_trials, "number of iterations");
        parser.set_optional<std::string>("name_format",
                                         "name_format",
                                         "json",
                                         "either json, human, or txt");

        // Optionally run an evenly split subset of benchmarks for autotuning.
        parser.set_optional<int>("parallel_instance",
                                 "parallel_instance",
                                 0,
                                 "parallel instance index");
        parser.set_optional<int>("parallel_instances",
                                 "parallel_instances",
                                 1,
                                 "total parallel instances");
    }

    void parse(cli::Parser& parser)
    {
        size = parser.get<size_t>("size");

        seed_type = parser.get<std::string>("seed");

        seed = managed_seed(seed_type);

        batch_iterations  = parser.get<size_t>("batch_iterations");
        warmup_iterations = parser.get<size_t>("warmup_iterations");

        cold            = !parser.get<bool>("hot");
        record_as_whole = parser.get<bool>("record_as_whole");

        trials             = parser.get<int>("trials");
        parallel_instance  = parser.get<int>("parallel_instance");
        parallel_instances = parser.get<int>("parallel_instances");

        bench_naming::set_format(parser.get<std::string>("name_format"));
    }

    void add_context()
    {
        benchmark::AddCustomContext("size", std::to_string(size));
        benchmark::AddCustomContext("seed", seed_type);

        benchmark::AddCustomContext("batch_iterations", std::to_string(batch_iterations));
        benchmark::AddCustomContext("warmup_iterations", std::to_string(warmup_iterations));

        hipDeviceProp_t devProp;
        int             device_id = 0;
        HIP_CHECK(hipGetDevice(&device_id));
        HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));

        auto str = [](const std::string& name, const std::string& val)
        { benchmark::AddCustomContext(name, val); };

        auto num = [](const std::string& name, const auto& value)
        { benchmark::AddCustomContext(name, std::to_string(value)); };

        auto dim2 = [num](const std::string& name, const auto* values)
        {
            num(name + "_x", values[0]);
            num(name + "_y", values[1]);
        };

        auto dim3 = [num, dim2](const std::string& name, const auto* values)
        {
            dim2(name, values);
            num(name + "_z", values[2]);
        };

        str("hdp_name", devProp.name);
        num("hdp_total_global_mem", devProp.totalGlobalMem);
        num("hdp_shared_mem_per_block", devProp.sharedMemPerBlock);
        num("hdp_regs_per_block", devProp.regsPerBlock);
        num("hdp_warp_size", devProp.warpSize);
        num("hdp_max_threads_per_block", devProp.maxThreadsPerBlock);
        dim3("hdp_max_threads_dim", devProp.maxThreadsDim);
        dim3("hdp_max_grid_size", devProp.maxGridSize);
        num("hdp_clock_rate", devProp.clockRate);
        num("hdp_memory_clock_rate", devProp.memoryClockRate);
        num("hdp_memory_bus_width", devProp.memoryBusWidth);
        num("hdp_total_const_mem", devProp.totalConstMem);
        num("hdp_major", devProp.major);
        num("hdp_minor", devProp.minor);
        num("hdp_multi_processor_count", devProp.multiProcessorCount);
        num("hdp_l2_cache_size", devProp.l2CacheSize);
        num("hdp_max_threads_per_multiprocessor", devProp.maxThreadsPerMultiProcessor);
        num("hdp_compute_mode", devProp.computeMode);
        num("hdp_clock_instruction_rate", devProp.clockInstructionRate);
        num("hdp_concurrent_kernels", devProp.concurrentKernels);
        num("hdp_pci_domain_id", devProp.pciDomainID);
        num("hdp_pci_bus_id", devProp.pciBusID);
        num("hdp_pci_device_id", devProp.pciDeviceID);
        num("hdp_max_shared_memory_per_multi_processor", devProp.maxSharedMemoryPerMultiProcessor);
        num("hdp_is_multi_gpu_board", devProp.isMultiGpuBoard);
        num("hdp_can_map_host_memory", devProp.canMapHostMemory);
        str("hdp_gcn_arch_name", devProp.gcnArchName);
        num("hdp_integrated", devProp.integrated);
        num("hdp_cooperative_launch", devProp.cooperativeLaunch);
        num("hdp_cooperative_multi_device_launch", devProp.cooperativeMultiDeviceLaunch);
        num("hdp_max_texture_1d_linear", devProp.maxTexture1DLinear);
        num("hdp_max_texture_1d", devProp.maxTexture1D);
        dim2("hdp_max_texture_2d", devProp.maxTexture2D);
        dim3("hdp_max_texture_3d", devProp.maxTexture3D);
        num("hdp_mem_pitch", devProp.memPitch);
        num("hdp_texture_alignment", devProp.textureAlignment);
        num("hdp_texture_pitch_alignment", devProp.texturePitchAlignment);
        num("hdp_kernel_exec_timeout_enabled", devProp.kernelExecTimeoutEnabled);
        num("hdp_ecc_enabled", devProp.ECCEnabled);
        num("hdp_tcc_driver", devProp.tccDriver);
        num("hdp_cooperative_multi_device_unmatched_func",
            devProp.cooperativeMultiDeviceUnmatchedFunc);
        num("hdp_cooperative_multi_device_unmatched_grid_dim",
            devProp.cooperativeMultiDeviceUnmatchedGridDim);
        num("hdp_cooperative_multi_device_unmatched_block_dim",
            devProp.cooperativeMultiDeviceUnmatchedBlockDim);
        num("hdp_cooperative_multi_device_unmatched_shared_mem",
            devProp.cooperativeMultiDeviceUnmatchedSharedMem);
        num("hdp_is_large_bar", devProp.isLargeBar);
        num("hdp_asic_revision", devProp.asicRevision);
        num("hdp_managed_memory", devProp.managedMemory);
        num("hdp_direct_managed_mem_access_from_host", devProp.directManagedMemAccessFromHost);
        num("hdp_concurrent_managed_access", devProp.concurrentManagedAccess);
        num("hdp_pageable_memory_access", devProp.pageableMemoryAccess);
        num("hdp_pageable_memory_access_uses_host_page_tables",
            devProp.pageableMemoryAccessUsesHostPageTables);

        const auto arch = devProp.arch;
        num("hdp_arch_has_global_int32_atomics", arch.hasGlobalInt32Atomics);
        num("hdp_arch_has_global_float_atomic_exch", arch.hasGlobalFloatAtomicExch);
        num("hdp_arch_has_shared_int32_atomics", arch.hasSharedInt32Atomics);
        num("hdp_arch_has_shared_float_atomic_exch", arch.hasSharedFloatAtomicExch);
        num("hdp_arch_has_float_atomic_add", arch.hasFloatAtomicAdd);
        num("hdp_arch_has_global_int64_atomics", arch.hasGlobalInt64Atomics);
        num("hdp_arch_has_shared_int64_atomics", arch.hasSharedInt64Atomics);
        num("hdp_arch_has_doubles", arch.hasDoubles);
        num("hdp_arch_has_warp_vote", arch.hasWarpVote);
        num("hdp_arch_has_warp_ballot", arch.hasWarpBallot);
        num("hdp_arch_has_warp_shuffle", arch.hasWarpShuffle);
        num("hdp_arch_has_funnel_shift", arch.hasFunnelShift);
        num("hdp_arch_has_thread_fence_system", arch.hasThreadFenceSystem);
        num("hdp_arch_has_sync_threads_ext", arch.hasSyncThreadsExt);
        num("hdp_arch_has_surface_funcs", arch.hasSurfaceFuncs);
        num("hdp_arch_has_3d_grid", arch.has3dGrid);
        num("hdp_arch_has_dynamic_parallelism", arch.hasDynamicParallelism);
    }

    static std::vector<std::unique_ptr<autotune_interface>>& sorted_benchmarks()
    {
        static std::vector<std::unique_ptr<autotune_interface>> sorted_benchmarks;
        return sorted_benchmarks;
    }

    state new_state(benchmark::State& gbench_state)
    {
        return state(stream,
                     size,
                     seed,
                     batch_iterations,
                     gbench_state,
                     warmup_iterations,
                     cold,
                     record_as_whole);
    }

    void apply_settings(benchmark::internal::Benchmark* b)
    {
        b->UseManualTime();
        b->Unit(benchmark::kMillisecond);

        // trials is -1 by default.
        if(trials > 0)
        {
            b->Iterations(trials);
        }
    }

    // Register a subset of all benchmarks for the current parallel instance.
    void register_sorted_subset(int parallel_instance_index, int parallel_instance_count)
    {
        // Sort to get a consistent order, because the order of static variable initialization is undefined by the C++ standard.
        std::sort(sorted_benchmarks().begin(),
                  sorted_benchmarks().end(),
                  [](const auto& l, const auto& r) { return l->sort_key() < r->sort_key(); });

        size_t configs_per_instance
            = (sorted_benchmarks().size() + parallel_instance_count - 1) / parallel_instance_count;
        size_t start
            = std::min(parallel_instance_index * configs_per_instance, sorted_benchmarks().size());
        size_t end = std::min((parallel_instance_index + 1) * configs_per_instance,
                              sorted_benchmarks().size());

        for(size_t i = start; i < end; ++i)
        {
            autotune_interface* benchmark = sorted_benchmarks().at(i).get();

            apply_settings(benchmark::RegisterBenchmark(
                benchmark->name().c_str(),
                [=](benchmark::State& gbench_state) { benchmark->run(new_state(gbench_state)); }));
        }
    }

    hipStream_t  stream = hipStreamDefault;
    size_t       size;
    std::string  seed_type;
    managed_seed seed;
    size_t       batch_iterations;
    size_t       warmup_iterations;
    bool         cold;
    bool         record_as_whole;

    int trials;
    int parallel_instance;
    int parallel_instances;
};

} // namespace benchmark_utils

#endif // ROCPRIM_BENCHMARK_UTILS_HPP_

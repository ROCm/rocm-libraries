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

#ifdef BENCHMARK_USE_AMDSMI
    #include <amd_smi/amdsmi.h>
#endif

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

#ifdef BENCHMARK_USE_AMDSMI
class amdsmi
{
public:
    amdsmi()
    {
        amdsmi_init(AMDSMI_INIT_AMD_GPUS);

        // These can't be turned into a member initializer list,
        // because the amdsmi_init() above has to be called first.
        m_target  = get_target();
        m_context = get_context(m_target);
    }

    ~amdsmi()
    {
        amdsmi_shut_down();
    }

    struct stats
    {
        amdsmi_gpu_metrics_t metrics;

        // Clocks (current frequencies in MHz)
        std::map<std::string, std::optional<uint32_t>> clocks;

        std::optional<uint64_t> vram_used_bytes;
    };

    std::string serialize_stats(const stats& stats) const
    {
        std::ostringstream ss;
        ss << "{";

        ss << "\"metrics\":" << serialize_metrics(stats.metrics);

        ss << ",\"clocks\":{";
        bool first = true;
        for(const auto& kv : stats.clocks)
        {
            if(!first)
                ss << ",";
            ss << "\"" << kv.first << "\":" << serialize_optional(kv.second);
            first = false;
        }
        ss << "}";

        ss << ",\"vram_used_bytes\":" << serialize_optional(stats.vram_used_bytes);

        ss << "}";
        return ss.str();
    }

    std::string serialize_context() const
    {
        const auto& ctx = m_context;

        std::ostringstream ss;
        ss << "{";

        ss << "\"product_name\":" << serialize_optional_string(ctx.product_name);
        ss << ",\"amdsmi_version\":" << serialize_optional_string(ctx.amdsmi_version);

        ss << ",\"amdsmi_metrics_version\":{";
        ss << "\"format_revision\":" << std::to_string(ctx.amdsmi_metrics_version.format_revision);
        ss << ",\"content_revision\":"
           << std::to_string(ctx.amdsmi_metrics_version.content_revision);
        ss << "}";

        // Clocks min/max
        ss << ",\"clocks\":{";
        bool first = true;
        for(const auto& kv : ctx.clocks)
        {
            if(!first)
                ss << ",";
            ss << "\"" << kv.first << "\":";
            if(kv.second)
                ss << "{"
                   << "\"min_clk\":" << kv.second->first << ",\"max_clk\":" << kv.second->second
                   << "}";
            else
                ss << "null";
            first = false;
        }
        ss << "}";

        // Power cap info
        ss << ",\"power_cap\":" << serialize_optional(ctx.power_cap);
        ss << ",\"power_cap_default\":" << serialize_optional(ctx.power_cap_default);
        ss << ",\"power_cap_dpm\":" << serialize_optional(ctx.power_cap_dpm);

        ss << ",\"energy_resolution\":" << serialize_optional(ctx.energy_resolution);

        // VRAM vendor + total
        ss << ",\"vram_vendor\":" << serialize_optional_string(ctx.vram_vendor);
        ss << ",\"vram_total_bytes\":" << serialize_optional(ctx.vram_total_bytes);

        // Stats
        ss << ",\"stats\":" << serialize_stats(ctx.stats);

        ss << "}";
        return ss.str();
    }

    stats get_stats() const
    {
        stats stats{};

        // Copy all GPU metrics
        amdsmi_gpu_metrics_t metrics{};
        if(amdsmi_get_gpu_metrics_info(m_target, &metrics) == AMDSMI_STATUS_SUCCESS)
            stats.metrics = metrics;

        // Clocks
        for(auto clk : clk_types)
        {
            amdsmi_clk_info_t clk_info{};
            if(amdsmi_get_clock_info(m_target, clk, &clk_info) == AMDSMI_STATUS_SUCCESS)
                stats.clocks[clk_type_to_string(clk)] = clk_info.clk;
        }

        // Memory usage
        uint64_t vram_used;
        if(amdsmi_get_gpu_memory_usage(m_target, AMDSMI_MEM_TYPE_VRAM, &vram_used)
           == AMDSMI_STATUS_SUCCESS)
            stats.vram_used_bytes = vram_used;

        return stats;
    }

private:
    struct context
    {
        std::optional<std::string> product_name;

        std::string amdsmi_version;

        struct
        {
            uint8_t format_revision;
            uint8_t content_revision;
        } amdsmi_metrics_version;

        // Clocks (min/max in MHz)
        std::map<std::string, std::optional<std::pair<uint32_t, uint32_t>>> clocks;

        std::optional<uint64_t> power_cap;
        std::optional<uint64_t> power_cap_default;
        std::optional<uint64_t> power_cap_dpm;

        std::optional<float> energy_resolution;

        std::optional<std::string> vram_vendor;
        std::optional<uint64_t>    vram_total_bytes;

        stats stats;
    };

    const std::vector<amdsmi_clk_type_t> clk_types = {
        AMDSMI_CLK_TYPE_SYS,
        AMDSMI_CLK_TYPE_DF,
        AMDSMI_CLK_TYPE_DCEF,
        AMDSMI_CLK_TYPE_SOC,
        AMDSMI_CLK_TYPE_MEM,
        AMDSMI_CLK_TYPE_PCIE,
        AMDSMI_CLK_TYPE_VCLK0,
        AMDSMI_CLK_TYPE_VCLK1,
        AMDSMI_CLK_TYPE_DCLK0,
        AMDSMI_CLK_TYPE_DCLK1,
    };

    amdsmi_processor_handle get_target() const
    {
        // Get HIP device
        hipStream_t stream;
        HIP_CHECK(hipStreamCreate(&stream));
        int hip_dev;
        HIP_CHECK(hipStreamGetDevice(stream, &hip_dev));
        char hip_bus_id[64];
        HIP_CHECK(hipDeviceGetPCIBusId(hip_bus_id, sizeof(hip_bus_id), hip_dev));

        // Get all AMD SMI sockets
        uint32_t socket_count;
        amdsmi_get_socket_handles(&socket_count, nullptr);
        std::vector<amdsmi_socket_handle> sockets(socket_count);
        amdsmi_get_socket_handles(&socket_count, sockets.data());

        // Get AMD SMI GPU device handle of the HIP device
        amdsmi_processor_handle target = nullptr;
        for(auto s : sockets)
        {
            uint32_t dev_count = 0;
            amdsmi_get_processor_handles(s, &dev_count, nullptr);
            std::vector<amdsmi_processor_handle> devs(dev_count);
            amdsmi_get_processor_handles(s, &dev_count, devs.data());
            for(auto h : devs)
            {
                processor_type_t type;
                amdsmi_get_processor_type(h, &type);
                if(type != AMDSMI_PROCESSOR_TYPE_AMD_GPU)
                    continue;
                if(get_amdsmi_pci_bus_id(h) == hip_bus_id)
                {
                    target = h;
                    break;
                }
            }
            if(target)
                break;
        }

        if(!target)
        {
            std::cerr << "No matching GPU found for HIP device " << hip_dev << "\n";
            exit(1);
        }

        return target;
    }

    std::string get_amdsmi_pci_bus_id(amdsmi_processor_handle h) const
    {
        uint64_t bdfid;
        amdsmi_get_gpu_bdf_id(h, &bdfid);
        uint32_t           domain   = (bdfid >> 32) & 0xffff;
        uint32_t           bus      = (bdfid >> 8) & 0xff;
        uint32_t           device   = (bdfid >> 3) & 0x1f;
        uint32_t           function = bdfid & 0x7;
        std::ostringstream oss;
        oss << std::setfill('0') << std::hex << std::setw(4) << domain << ":" << std::setw(2) << bus
            << ":" << std::setw(2) << device << "." << function;
        return oss.str();
    }

    context get_context(amdsmi_processor_handle target) const
    {
        context ctx;

        amdsmi_board_info_t board_info;
        if(amdsmi_get_gpu_board_info(target, &board_info) == AMDSMI_STATUS_SUCCESS)
            ctx.product_name = board_info.product_name;

        amdsmi_version_t amdsmi_version;
        if(amdsmi_get_lib_version(&amdsmi_version) == AMDSMI_STATUS_SUCCESS)
            ctx.amdsmi_version = amdsmi_version.build;

        amdsmi_gpu_metrics_t metrics{};
        if(amdsmi_get_gpu_metrics_info(target, &metrics) == AMDSMI_STATUS_SUCCESS)
            ctx.amdsmi_metrics_version.format_revision = metrics.common_header.format_revision;
        ctx.amdsmi_metrics_version.content_revision = metrics.common_header.content_revision;

        for(auto clk : clk_types)
        {
            amdsmi_clk_info_t clk_info{};
            if(amdsmi_get_clock_info(target, clk, &clk_info) == AMDSMI_STATUS_SUCCESS)
                ctx.clocks[clk_type_to_string(clk)]
                    = std::make_pair(clk_info.min_clk, clk_info.max_clk);
        }

        amdsmi_power_cap_info_t pcap;
        if(amdsmi_get_power_cap_info(target, 0, &pcap) == AMDSMI_STATUS_SUCCESS)
        {
            ctx.power_cap         = pcap.power_cap;
            ctx.power_cap_default = pcap.default_power_cap;
            ctx.power_cap_dpm     = pcap.dpm_cap;
        }

        uint64_t energy_acc, energy_ts;
        float    energy_res;
        if(amdsmi_get_energy_count(target, &energy_acc, &energy_res, &energy_ts)
           == AMDSMI_STATUS_SUCCESS)
            ctx.energy_resolution = energy_res;

        char vram_vendor_buf[128];
        if(amdsmi_get_gpu_vram_vendor(target, vram_vendor_buf, sizeof(vram_vendor_buf))
           == AMDSMI_STATUS_SUCCESS)
            ctx.vram_vendor = vram_vendor_buf;

        uint64_t vram_total;
        if(amdsmi_get_gpu_memory_total(target, AMDSMI_MEM_TYPE_VRAM, &vram_total)
           == AMDSMI_STATUS_SUCCESS)
            ctx.vram_total_bytes = vram_total;

        ctx.stats = get_stats();

        return ctx;
    }

    std::string clk_type_to_string(amdsmi_clk_type_t clk) const
    {
        switch(clk)
        {
            case AMDSMI_CLK_TYPE_SYS: return "sys";
            case AMDSMI_CLK_TYPE_DF: return "df";
            case AMDSMI_CLK_TYPE_DCEF: return "dcef";
            case AMDSMI_CLK_TYPE_SOC: return "soc";
            case AMDSMI_CLK_TYPE_MEM: return "mem";
            case AMDSMI_CLK_TYPE_PCIE: return "pcie";
            case AMDSMI_CLK_TYPE_VCLK0: return "vclk0";
            case AMDSMI_CLK_TYPE_VCLK1: return "vclk1";
            case AMDSMI_CLK_TYPE_DCLK0: return "dclk0";
            case AMDSMI_CLK_TYPE_DCLK1: return "dclk1";
        }
        std::cerr << "Failed to match clock type " << clk << " to a string\n";
        exit(1);
    }

    std::string serialize_optional_string(const std::optional<std::string>& opt) const
    {
        return opt ? ("\"" + *opt + "\"") : "null";
    }

    template<typename T>
    std::string serialize_optional(const std::optional<T>& opt) const
    {
        return opt ? std::to_string(*opt) : "null";
    }

    std::string serialize_metrics(const amdsmi_gpu_metrics_t& metrics) const
    {
        std::ostringstream ss;
        ss << "{";

        bool first     = true;
        auto add_comma = [&]()
        {
            if(!first)
                ss << ",";
            first = false;
        };

        auto add_field = [&](const char* name, auto value)
        {
            add_comma();
            ss << "\"" << name << "\":" << value;
        };

        auto add_array = [&](const char* name, auto&& arr)
        {
            add_comma();
            ss << "\"" << name << "\":[";
            for(size_t i = 0; i < (sizeof(arr) / sizeof(*arr)); ++i)
            {
                if(i > 0)
                    ss << ",";
                ss << arr[i];
            }
            ss << "]";
        };

    // Convenience macros to avoid repeating field names
    #define ADD(field) add_field(#field, metrics.field)
    #define ADD_ARRAY(field) add_array(#field, metrics.field)

        struct revision_block
        {
            int                   min_content_revision;
            std::function<void()> serialize;
        };

        std::vector<revision_block> blocks = {
            {0,
             [&]
             {
             ADD(temperature_edge);
             ADD(temperature_hotspot);
             ADD(temperature_mem);
             ADD(temperature_vrgfx);
             ADD(temperature_vrsoc);
             ADD(temperature_vrmem);

             ADD(average_gfx_activity);
             ADD(average_umc_activity);
             ADD(average_mm_activity);

             ADD(average_socket_power);
             ADD(energy_accumulator);
             ADD(system_clock_counter);

             ADD(average_gfxclk_frequency);
             ADD(average_socclk_frequency);
             ADD(average_uclk_frequency);
             ADD(average_vclk0_frequency);
             ADD(average_dclk0_frequency);
             ADD(average_vclk1_frequency);
             ADD(average_dclk1_frequency);

             ADD(current_gfxclk);
             ADD(current_socclk);
             ADD(current_uclk);
             ADD(current_vclk0);
             ADD(current_dclk0);
             ADD(current_vclk1);
             ADD(current_dclk1);

             ADD(throttle_status);
             ADD(current_fan_speed);
             ADD(pcie_link_width);
             ADD(pcie_link_speed);
             }                                  },
            {1,
             [&]
             {
             ADD(gfx_activity_acc);
             ADD(mem_activity_acc);
             ADD_ARRAY(temperature_hbm);
             }                                  },
            {2, [&] { ADD(firmware_timestamp); }},
            {3,
             [&]
             {
             ADD(voltage_soc);
             ADD(voltage_gfx);
             ADD(voltage_mem);
             ADD(indep_throttle_status);
             }                                  },
            {4,
             [&]
             {
             ADD(current_socket_power);
             ADD_ARRAY(vcn_activity);
             ADD(gfxclk_lock_status);
             ADD(xgmi_link_width);
             ADD(xgmi_link_speed);
             ADD(pcie_bandwidth_acc);
             ADD(pcie_bandwidth_inst);
             ADD(pcie_l0_to_recov_count_acc);
             ADD(pcie_replay_count_acc);
             ADD(pcie_replay_rover_count_acc);
             ADD_ARRAY(xgmi_read_data_acc);
             ADD_ARRAY(xgmi_write_data_acc);
             ADD_ARRAY(current_gfxclks);
             ADD_ARRAY(current_socclks);
             ADD_ARRAY(current_vclk0s);
             ADD_ARRAY(current_dclk0s);
             }                                  },
            {5,
             [&]
             {
             ADD_ARRAY(jpeg_activity);
             ADD(pcie_nak_sent_count_acc);
             ADD(pcie_nak_rcvd_count_acc);
             }                                  },
            {6,
             [&]
             {
             ADD(accumulation_counter);
             ADD(prochot_residency_acc);
             ADD(ppt_residency_acc);
             ADD(socket_thm_residency_acc);
             ADD(vr_thm_residency_acc);
             ADD(hbm_thm_residency_acc);
             ADD(num_partition);
             // xcp_stats is too annoying and unimportant to serialize.
             ADD(pcie_lc_perf_other_end_recovery);
             }                                  },
            {7,
             [&]
             {
             ADD(vram_max_bandwidth);
             ADD_ARRAY(xgmi_link_status);
             }                                  },
        };

        int currentRev = m_context.amdsmi_metrics_version.content_revision;
        for(auto& block : blocks)
        {
            if(currentRev < block.min_content_revision)
                break;
            block.serialize();
        }

    #undef ADD
    #undef ADD_ARRAY

        ss << "}";
        return ss.str();
    }

    amdsmi_processor_handle m_target;
    context                 m_context;
};
#endif

class logger
{
public:
    logger(std::string iteration_info_out)
        : m_iteration_info_out(iteration_info_out), m_is_active(!iteration_info_out.empty())
    {}

#ifdef BENCHMARK_USE_AMDSMI
    void record_amdsmi_stats()
    {
        if(!m_is_active)
            return;

        m_amdsmi_stats = m_amdsmi.get_stats();
    }
#endif

    void save_batch_times(const std::vector<double> batch_times)
    {
        if(!m_is_active)
            return;

        struct batch batch = {};

        batch.batch_times = batch_times;

#ifdef BENCHMARK_USE_AMDSMI
        batch.amdsmi_stats = m_amdsmi_stats;
#endif

        m_batches.emplace_back(batch);
    }

    // This function doesn't use a JSON library,
    // since it just appends text onto the end of a JSON file.
    // Seeking and writing to the end of a huge file is very fast.
    void output_specialization_info(const std::string& name) const
    {
        if(!m_is_active)
            return;

        // Try opening file
        std::fstream outfile(m_iteration_info_out, std::ios::in | std::ios::out | std::ios::ate);
        bool         new_file = false;

        if(!outfile)
        {
            // File doesn’t exist, create it
            outfile.open(m_iteration_info_out, std::ios::out | std::ios::trunc);
            new_file = true;
        }
        else
        {
            // File exists, check size
            outfile.seekp(0, std::ios::end);
            auto end_pos = outfile.tellp();
            if(end_pos == 0)
            {
                new_file = true;
            }
            else if(end_pos < 2)
            {
                std::cerr << "Malformed JSON file.\n";
                std::exit(EXIT_FAILURE);
            }
            else
            {
                // Trim trailing "]}" so we can append new benchmarks
                outfile.seekp(-2, std::ios::end);
            }
        }

        if(new_file)
        {
            outfile << "{";
            outfile << "\"context\":{";

#ifdef BENCHMARK_USE_AMDSMI
            outfile << "\"amdsmi\":" << m_amdsmi.serialize_context();
#endif

            outfile << "},";
            outfile << "\"benchmarks\":[";
        }
        else
        {
            outfile << ",";
        }

        // Append the serialized benchmark
        outfile << serialize_benchmark(name);

        // Close benchmarks array and JSON
        outfile << "]}";
        outfile.close();
    }

private:
    struct batch
    {
        std::vector<double> batch_times;

#ifdef BENCHMARK_USE_AMDSMI
        amdsmi::stats amdsmi_stats;
#endif
    };

    std::string serialize_benchmark(const std::string& name) const
    {
        std::ostringstream ss;
        ss << "{";
        ss << "\"name\":\"" << name << "\",";
        ss << "\"batches\":[";
        for(size_t i = 0; i < m_batches.size(); i++)
        {
            if(i != 0)
                ss << ",";

            ss << "{";
            ss << "\"iterations_ms\":" << serialize_batch_times(m_batches[i].batch_times);

#ifdef BENCHMARK_USE_AMDSMI
            ss << ",\"amdsmi_stats_after_iterations\":"
               << m_amdsmi.serialize_stats(m_batches[i].amdsmi_stats);
#endif

            ss << "}";
        }
        ss << "]";
        ss << "}";
        return ss.str();
    }

    std::string serialize_batch_times(const std::vector<double>& batch_times) const
    {
        std::ostringstream ss;
        ss << "[";
        for(size_t i = 0; i < batch_times.size(); i++)
        {
            if(i != 0)
                ss << ",";
            ss << batch_times[i];
        }
        ss << "]";
        return ss.str();
    }

    std::string m_iteration_info_out;
    bool        m_is_active;

    std::vector<batch> m_batches;

#ifdef BENCHMARK_USE_AMDSMI
    amdsmi        m_amdsmi;
    amdsmi::stats m_amdsmi_stats;
#endif
};

class state
{
public:
    state(hipStream_t         stream,
          size_t              bytes,
          const managed_seed& seed,
          size_t              batch_iterations,
          benchmark::State&   gbench_state,
          size_t              warmup_iterations,
          bool                cold,
          std::string         iteration_info_out)
        : stream(stream)
        , bytes(bytes)
        , seed(seed)
        , batch_iterations(batch_iterations)
        , gbench_state(gbench_state)
        , m_warmup_iterations(warmup_iterations)
        , m_cold(cold)
        , m_events(batch_iterations * 2)
        , m_logger(iteration_info_out)
    {
        for(auto& event : m_events)
        {
            HIP_CHECK(hipEventCreate(&event));
        }
    }

    ~state()
    {
        for(const auto& event : m_events)
        {
            HIP_CHECK(hipEventDestroy(event));
        }
    }

    // Used to reset the input array of algorithms like device_merge_inplace.
    void run_before_every_iteration(std::function<void()> lambda)
    {
        m_run_before_every_iteration_lambda = lambda;
    }

    // Used to accumulate the results of state.run() calls.
    void accumulate_total_gbench_iterations_every_run()
    {
        m_reset_total_gbench_iterations_every_run = false;
    }

    void run(std::function<void()> kernel)
    {
        // Warm-up
        for(size_t i = 0; i < m_warmup_iterations; ++i)
        {
            // Benchmarks may expect their kernel input to be prepared by this lambda,
            // so to prevent any potential crashes, we call the lambda during warm-up.
            if(m_run_before_every_iteration_lambda)
            {
                m_run_before_every_iteration_lambda();
            }

            kernel();
        }
        HIP_CHECK(hipDeviceSynchronize());

        // Run
        for(auto _ : gbench_state)
        {
            for(size_t i = 0; i < batch_iterations; ++i)
            {
                if(m_run_before_every_iteration_lambda)
                {
                    m_run_before_every_iteration_lambda();
                }

                if(m_cold)
                {
                    clear_gpu_cache(stream);
                }

                // Even m_events record the start time.
                HIP_CHECK(hipEventRecord(m_events[i * 2], stream));

                kernel();

                // Odd m_events record the stop time.
                HIP_CHECK(hipEventRecord(m_events[i * 2 + 1], stream));
            }

            // Wait until the last record event has completed.
            HIP_CHECK(hipEventSynchronize(m_events[batch_iterations * 2 - 1]));

#ifdef BENCHMARK_USE_AMDSMI
            // This is a *really* slow call, but you can increase batch_iterations
            // if this ever dominates the benchmarking runtime.
            m_logger.record_amdsmi_stats();
#endif

            std::vector<double> batch_times;

            // Accumulate the total elapsed time.
            double elapsed_ms = 0.0;
            for(size_t i = 0; i < batch_iterations; i++)
            {
                float iteration_ms;
                HIP_CHECK(hipEventElapsedTime(&iteration_ms, m_events[i * 2], m_events[i * 2 + 1]));
                m_times.emplace_back(iteration_ms);
                batch_times.emplace_back(iteration_ms);
                elapsed_ms += iteration_ms;
            }

            gbench_state.SetIterationTime(elapsed_ms / 1000);

            m_logger.save_batch_times(batch_times);

            // TODO: When gbench is removed in the future, replace the gbench_state loop
            // with an infinite while-loop, and use this instead:
            // now = time.now()
            // elapsed = now - start
            // if elapsed > min_time {
            //     break
            // }
        }

        if(m_reset_total_gbench_iterations_every_run)
        {
            m_total_gbench_iterations = 0;
        }
        m_total_gbench_iterations += gbench_state.iterations();
    }

    void set_throughput(size_t actual_size, size_t type_size)
    {
        if(m_has_set_throughput)
        {
            std::cerr << "Error: Benchmarks should only ever call set_throughput() once, at the "
                         "very end.\n";
            exit(EXIT_FAILURE);
        }
        m_has_set_throughput = true;

        gbench_state.SetBytesProcessed(m_total_gbench_iterations * batch_iterations * actual_size
                                       * type_size);
        gbench_state.SetItemsProcessed(m_total_gbench_iterations * batch_iterations * actual_size);

        output_statistics();

        std::string name = get_escaped_name(gbench_state.name());
        m_logger.output_specialization_info(name);
    }

    // These are directly read by benchmarks.
    hipStream_t       stream;
    size_t            bytes;
    managed_seed      seed;
    size_t            batch_iterations;
    benchmark::State& gbench_state;

private:
    // Zeros a 256 MiB buffer, used to clear the cache before each kernel call.
    // 256 MiB is the size of the largest cache on any AMD GPU.
    // It is currently not possible to fetch the L3 cache size from the runtime.
    void clear_gpu_cache(hipStream_t stream) const
    {
        constexpr size_t buf_size = 256 * MiB;
        static void*     buf      = nullptr;
        if(!buf)
        {
            HIP_CHECK(hipMalloc(&buf, buf_size));
        }
        HIP_CHECK(hipMemsetAsync(buf, 0, buf_size, stream));
    }

    std::string get_escaped_name(const std::string& name) const
    {
        std::string escaped_name;
        for(char c : name)
        {
            if(c == '"' || c == '\\')
                escaped_name += '\\';
            escaped_name += c;
        }
        escaped_name += "/manual_time";
        return escaped_name;
    }

    void output_statistics() const
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

    double get_mean() const
    {
        return std::reduce(m_times.begin(), m_times.end()) / m_times.size();
    }

    double get_median() const
    {
        auto tmp = m_times; // Copy, so we don’t mutate *this.
        std::sort(tmp.begin(), tmp.end());

        size_t n = tmp.size();
        if(n % 2 == 1)
        {
            // Middle element.
            return tmp[n / 2];
        }
        else
        {
            // Average of two middle elements.
            return (tmp[n / 2 - 1] + tmp[n / 2]) / 2.0;
        }
    }

    double get_stddev(double mean) const
    {
        auto SumSquares = [](const std::vector<double>& v)
        { return std::transform_reduce(v.begin(), v.end(), v.begin(), 0.0); };
        auto Sqr  = [](double dat) { return dat * dat; };
        auto Sqrt = [](double dat) { return dat < 0.0 ? 0.0 : std::sqrt(dat); };

        double stddev = 0.0;
        if(m_times.size() > 1)
        {
            double avg_squares = SumSquares(m_times) * (1.0 / m_times.size());
            stddev = Sqrt(m_times.size() / (m_times.size() - 1.0) * (avg_squares - Sqr(mean)));
        }
        return stddev;
    }

    double get_cv(double stddev, double mean) const
    {
        return m_times.size() >= 2 ? stddev / mean : 0.0;
    }

    size_t m_warmup_iterations;
    bool   m_cold;

    std::vector<hipEvent_t> m_events;
    logger                  m_logger;
    std::function<void()>   m_run_before_every_iteration_lambda       = nullptr;
    size_t                  m_total_gbench_iterations                 = 0;
    bool                    m_reset_total_gbench_iterations_every_run = true;
    std::vector<double>     m_times;
    bool                    m_has_set_throughput = false;
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
        register_sorted_subset(m_parallel_instance, m_parallel_instances);
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

        parser.set_optional<std::string>(
            "iteration_info_out",
            "iteration_info_out",
            "",
            "optional output path for a JSON file containing iteration info");
    }

    void parse(cli::Parser& parser)
    {
        m_bytes = parser.get<size_t>("size");

        m_seed_type = parser.get<std::string>("seed");
        m_seed      = managed_seed(m_seed_type);

        m_batch_iterations  = parser.get<size_t>("batch_iterations");
        m_warmup_iterations = parser.get<size_t>("warmup_iterations");

        m_cold = !parser.get<bool>("hot");

        m_iteration_info_out = parser.get<std::string>("iteration_info_out");

        m_trials             = parser.get<int>("trials");
        m_parallel_instance  = parser.get<int>("parallel_instance");
        m_parallel_instances = parser.get<int>("parallel_instances");

        bench_naming::set_format(parser.get<std::string>("name_format"));
    }

    void add_context()
    {
        benchmark::AddCustomContext("size", std::to_string(m_bytes));
        benchmark::AddCustomContext("seed", m_seed_type);

        benchmark::AddCustomContext("batch_iterations", std::to_string(m_batch_iterations));
        benchmark::AddCustomContext("warmup_iterations", std::to_string(m_warmup_iterations));

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
        return state(m_stream,
                     m_bytes,
                     m_seed,
                     m_batch_iterations,
                     gbench_state,
                     m_warmup_iterations,
                     m_cold,
                     m_iteration_info_out);
    }

    void apply_settings(benchmark::internal::Benchmark* b)
    {
        b->UseManualTime();
        b->Unit(benchmark::kMillisecond);

        // trials is -1 by default.
        if(m_trials > 0)
        {
            b->Iterations(m_trials);
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

    hipStream_t  m_stream = hipStreamDefault;
    size_t       m_bytes;
    std::string  m_seed_type;
    managed_seed m_seed;
    size_t       m_batch_iterations;
    size_t       m_warmup_iterations;
    bool         m_cold;
    std::string  m_iteration_info_out;

    int m_trials;
    int m_parallel_instance;
    int m_parallel_instances;
};

} // namespace benchmark_utils

#endif // ROCPRIM_BENCHMARK_UTILS_HPP_

// Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../../common/utils.hpp"
#include "../../common/utils_custom_type.hpp"
#include "../../common/utils_data_generation.hpp"

#include "../common_test_header.hpp"

#include "test_seed.hpp"
#include "test_utils_custom_float_type.hpp"
#include "test_utils_custom_test_types.hpp"

#include <rocprim/config.hpp>
#include <rocprim/type_traits.hpp>
#include <rocprim/types.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <random>
#include <set>
#include <stdint.h>
#include <type_traits>
#include <utility>
#include <vector>

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<typename T>
struct numeric_limits_custom_test_type : public numeric_limits<typename T::value_type>
{};

} // namespace detail

// Numeric limits which also supports common::custom_type<T, T, true> and custom_test_array_type<T, N> classes
template<>
struct numeric_limits<test_utils::custom_float_type>
    : detail::numeric_limits_custom_test_type<test_utils::custom_float_type>
{};

template<typename T>
struct numeric_limits<common::custom_type<T, T, true>>
    : detail::numeric_limits_custom_test_type<common::custom_type<T, T, true>>
{};

template<typename T>
struct numeric_limits<test_utils::custom_non_copyable_type<T>>
    : detail::numeric_limits_custom_test_type<test_utils::custom_non_copyable_type<T>>
{};

template<typename T>
struct numeric_limits<test_utils::custom_non_moveable_type<T>>
    : detail::numeric_limits_custom_test_type<test_utils::custom_non_moveable_type<T>>
{};

template<typename T>
struct numeric_limits<test_utils::custom_non_default_type<T>>
    : detail::numeric_limits_custom_test_type<test_utils::custom_non_default_type<T>>
{};

template<typename T, size_t N>
struct numeric_limits<test_utils::custom_test_array_type<T, N>>
    : detail::numeric_limits_custom_test_type<test_utils::custom_test_array_type<T, N>>
{};

template<>
struct numeric_limits<half> : std::numeric_limits<half>
{
public:
    static inline half min()
    {
        return half(0.00006104f);
    };
    static inline half max()
    {
        return half(65504.0f);
    };
    static inline half lowest()
    {
        return half(-65504.0f);
    };
    static inline half infinity()
    {
        return half(std::numeric_limits<float>::infinity());
    };
    static inline half quiet_NaN()
    {
        return half(std::numeric_limits<float>::quiet_NaN());
    };
    static inline half signaling_NaN()
    {
        return half(std::numeric_limits<float>::signaling_NaN());
    };
};

template<>
struct numeric_limits<bfloat16> : std::numeric_limits<bfloat16>
{
public:
    static inline bfloat16 max()
    {
        return bfloat16(std::numeric_limits<float>::max() * 0.998);
    };
    static inline bfloat16 min()
    {
        return bfloat16(std::numeric_limits<float>::min());
    };
    static inline bfloat16 lowest()
    {
        return bfloat16(std::numeric_limits<float>::lowest() * 0.998);
    };
    static inline bfloat16 infinity()
    {
        return bfloat16(std::numeric_limits<float>::infinity());
    };
    static inline bfloat16 quiet_NaN()
    {
        return bfloat16(std::numeric_limits<float>::quiet_NaN());
    };
    static inline bfloat16 signaling_NaN()
    {
        return bfloat16(std::numeric_limits<float>::signaling_NaN());
    };
};
// End of extended numeric_limits

END_ROCPRIM_NAMESPACE

namespace common
{

template<class T>
struct generate_limits<T,
                       std::enable_if_t<test_utils::is_custom_test_array_type<T>::value
                                        || common::is_custom_type<T>::value>>
{
    using Type = typename T::value_type;
    static inline Type min()
    {
        return generate_limits<Type>::min();
    }
    static inline Type max()
    {
        return generate_limits<Type>::max();
    }
};

} // namespace common

namespace test_utils
{

static constexpr uint32_t random_data_generation_segments       = 32;
static constexpr uint32_t random_data_generation_repeat_strides = 4;

// Converts possible device side types to their relevant host side native types
inline rocprim::native_half convert_to_native(const rocprim::half& value)
{
    return rocprim::native_half(value);
}

inline rocprim::native_bfloat16 convert_to_native(const rocprim::bfloat16& value)
{
    return rocprim::native_bfloat16(value);
}

template<class T>
inline auto convert_to_native(const T& value)
{
    return value;
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
            std::vector<T> r = {rocprim::numeric_limits<T>::quiet_NaN(),
                                sign_bit_flip(rocprim::numeric_limits<T>::quiet_NaN()),
                                // TODO: switch on when signaling_NaN will be supported on NVIDIA
                                //rocprim::numeric_limits<T>::signaling_NaN(),
                                //sign_bit_flip(rocprim::numeric_limits<T>::signaling_NaN()),
                                rocprim::numeric_limits<T>::infinity(),
                                sign_bit_flip(rocprim::numeric_limits<T>::infinity()),
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
template<class OutputIter, class Generator>
void add_special_values(OutputIter it, const size_t size, Generator&& gen)
{
    using T                       = common::it_value_t<OutputIter>;
    std::vector<T> special_values = ::test_utils::special_values<T>::vector();
    if(size > special_values.size())
    {
        unsigned int start = gen() % (size - special_values.size());
        std::copy(special_values.begin(), special_values.end(), it + start);
    }
}

/// Safe sign-mixed comparisons, negative values always compare less
/// than any values of unsigned types (in contrast to the behaviour of the built-in comparison operator)
/// This is a backport of a C++20 standard library feature to C++14
template<class T, class U>
constexpr auto cmp_less(T t, U u) noexcept
    -> std::enable_if_t<std::is_signed<T>::value == std::is_signed<U>::value, bool>
{
    return t < u;
}

template<class T, class U>
constexpr auto cmp_less(T t, U u) noexcept
    -> std::enable_if_t<std::is_signed<T>::value && !std::is_signed<U>::value, bool>
{
    // U is unsigned
    return t < 0 || std::make_unsigned_t<T>(t) < u;
}

template<class T, class U>
constexpr auto cmp_less(T t, U u) noexcept
    -> std::enable_if_t<!std::is_signed<T>::value && std::is_signed<U>::value, bool>
{
    // T is unsigned U is signed
    return u >= 0 && t < std::make_unsigned_t<U>(u);
}

template<class T, class U>
constexpr bool cmp_greater(T t, U u) noexcept
{
    return cmp_less(u, t);
}

// Backport of saturate_cast from C++26 to C++14
// From https://github.com/llvm/llvm-project/blob/52b18430ae105566f26152c0efc63998301b1134/libcxx/include/__numeric/saturation_arithmetic.h#L97
// licensed under the MIT license
template<typename Res, typename T>
constexpr Res saturate_cast(T x) noexcept
{
    // Handle overflow
    if(cmp_less(x, rocprim::numeric_limits<Res>::min()))
    {
        return rocprim::numeric_limits<Res>::min();
    }
    if(cmp_greater(x, rocprim::numeric_limits<Res>::max()))
    {
        return rocprim::numeric_limits<Res>::max();
    }
    // No overflow
    return static_cast<Res>(x);
}

template<class OutputIter, class Generator>
inline OutputIter segmented_generate_n(OutputIter it, size_t size, Generator&& gen)
{
    const size_t segment_size = size / random_data_generation_segments;
    if(segment_size == 0)
    {
        return std::generate_n(it, size, std::move(gen));
    }

    for(uint32_t segment_index = 0; segment_index < random_data_generation_segments;
        segment_index++)
    {
        if(segment_index % random_data_generation_repeat_strides == 0)
        {
            const auto repeated_value = gen();
            std::fill(it + segment_size * segment_index,
                      it + segment_size * (segment_index + 1),
                      repeated_value);
        }
        else
        {
            std::generate_n(it + segment_size * segment_index, segment_size, gen);
        }
    }
    // Generate the remaining items
    std::generate_n(it + segment_size * random_data_generation_segments,
                    size - segment_size * random_data_generation_segments,
                    gen);
    return it + size;
}

template<typename T, typename U, typename V>
auto make_distribution(U min, V max)
{
    if constexpr(rocprim::is_floating_point<T>::value)
    {
        using dis_type = typename std::conditional<std::is_same<rocprim::half, T>::value
                                                       || std::is_same<rocprim::bfloat16, T>::value,
                                                   float,
                                                   T>::type;
        return std::uniform_real_distribution<dis_type>(static_cast<dis_type>(min),
                                                        static_cast<dis_type>(max));
    }
    else
    {
        using dist_t
            = std::conditional_t<common::is_valid_for_int_distribution<T>::value,
                                 T,
                                 std::conditional_t<std::is_signed<T>::value, int, unsigned int>>;

        return common::uniform_int_distribution<dist_t>(saturate_cast<dist_t>(min),
                                                        saturate_cast<dist_t>(max));
    }
}

template<class OutputIter, class U, class V, class Generator>
inline auto generate_random_data_n(OutputIter it, size_t size, U min, V max, Generator&& gen)
    -> std::enable_if_t<(rocprim::is_integral<common::it_value_t<OutputIter>>::value
                         || rocprim::is_floating_point<common::it_value_t<OutputIter>>::value)
                            && !common::is_custom_type<common::it_value_t<OutputIter>>::value,
                        OutputIter>
{
    using T = common::it_value_t<OutputIter>;

    auto distribution = make_distribution<T>(min, max);

    return segmented_generate_n(it, size, [&]() { return static_cast<T>(distribution(gen)); });
}

template<class OutputIter, class U, class V, class Generator>
inline auto generate_random_data_n(OutputIter it, size_t size, U min, V max, Generator&& gen)
    -> std::enable_if_t<
        std::is_same_v<common::it_value_t<OutputIter>, test_utils::custom_float_type>,
        OutputIter>
{
    using T       = common::it_value_t<OutputIter>;
    using value_t = typename T::value_type;

    value_t min_temp;
    value_t max_temp;

    if constexpr(common::is_custom_type<U>::value)
    {
        min_temp = min.x;
    }
    else
    {
        min_temp = min;
    }

    if constexpr(common::is_custom_type<V>::value)
    {
        max_temp = max.x;
    }
    else
    {
        max_temp = max;
    }

    auto distribution = make_distribution<value_t>(min_temp, max_temp);

    return segmented_generate_n(it,
                                size,
                                [&]() { return static_cast<value_t>(distribution(gen)); });
}

template<class OutputIter, class Generator>
inline auto generate_random_data_n(OutputIter                     it,
                                   size_t                         size,
                                   common::it_value_t<OutputIter> min,
                                   common::it_value_t<OutputIter> max,
                                   Generator&&                    gen)
    -> std::enable_if_t<
        common::is_custom_type<common::it_value_t<OutputIter>>::value
            && !(rocprim::is_integral<common::it_value_t<OutputIter>>::value
                 || rocprim::is_floating_point<common::it_value_t<OutputIter>>::value),
        OutputIter>
{
    using T        = common::it_value_t<OutputIter>;
    using first_t  = typename T::first_type;
    using second_t = typename T::second_type;

    auto first_dist  = make_distribution<first_t>(min.x, max.x);
    auto second_dist = make_distribution<second_t>(min.y, max.y);

    return segmented_generate_n(it,
                                size,
                                [&]() {
                                    return T(static_cast<first_t>(first_dist(gen)),
                                             static_cast<second_t>(second_dist(gen)));
                                });
}

template<class OutputIter, class Generator>
inline auto generate_random_data_n(OutputIter                                          it,
                                   size_t                                              size,
                                   typename common::it_value_t<OutputIter>::value_type min,
                                   typename common::it_value_t<OutputIter>::value_type max,
                                   Generator&&                                         gen)
    -> std::enable_if_t<
        is_custom_test_array_type<common::it_value_t<OutputIter>>::value
            && std::is_integral<typename common::it_value_t<OutputIter>::value_type>::value,
        OutputIter>
{
    using T = typename std::iterator_traits<OutputIter>::value_type;

    common::uniform_int_distribution<typename T::value_type> distribution(min, max);
    return std::generate_n(it,
                           size,
                           [&]()
                           {
                               T result;
                               for(size_t i = 0; i < T::size; i++)
                               {
                                   result.values[i] = distribution(gen);
                               }
                               return result;
                           });
}

template<class T, class U, class V>
inline std::vector<T> get_random_data(size_t size, U min, V max, seed_type seed_value)
{
    std::vector<T> data(size);
    engine_type    gen(seed_value);
    generate_random_data_n(data.begin(), size, min, max, gen);
    return data;
}

template<class T, class U, class V>
inline auto get_random_value(U min, V max, seed_type seed_value)
    -> std::enable_if_t<rocprim::is_arithmetic<T>::value, T>
{
    T           result;
    engine_type gen(seed_value);
    generate_random_data_n(&result, 1, min, max, gen);
    return result;
}

template<class T>
inline auto get_random_value(typename T::value_type min,
                             typename T::value_type max,
                             seed_type              seed_value)
    -> std::enable_if_t<common::is_custom_type<T>::value || is_custom_test_array_type<T>::value, T>
{
    typename T::value_type result;
    engine_type            gen(seed_value);
    generate_random_data_n(&result, 1, min, max, gen);
    return T{result};
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

template<class T>
std::vector<size_t> get_sizes(T seed_value)
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
    if(!common::use_hmm())
    {
        // hipMallocManaged() currently doesnt support zero byte allocation
        sizes.push_back(0);
    }

    const std::vector<size_t> random_sizes1 = get_random_data<size_t>(2, 2, 1 << 20, seed_value);
    sizes.insert(sizes.end(), random_sizes1.begin(), random_sizes1.end());

    const std::vector<size_t> random_sizes2 = get_random_data<size_t>(3, 2, 1 << 17, seed_value);
    sizes.insert(sizes.end(), random_sizes2.begin(), random_sizes2.end());

    std::sort(sizes.begin(), sizes.end());

    return sizes;
}

template<unsigned int MaxPow2 = 37, class T>
std::vector<size_t> get_large_sizes(T seed_value)
{
    std::vector<size_t> test_sizes = {
        (size_t{1} << 30) - 1,
        size_t{1} << 31,
        (size_t{1} << 32) - 15,
        (size_t{1} << 33) + (size_t{1} << 32) - 876543,
        (size_t{1} << 34) - 12346,
        (size_t{1} << 35) + 1,
        (size_t{1} << MaxPow2) - 1,
    };
    const std::vector<size_t> random_sizes
        = get_random_data<size_t>(2, (size_t{1} << 30) + 1, (size_t{1} << MaxPow2) - 2, seed_value);

    std::vector<size_t> sizes(test_sizes.size() + random_sizes.size());
    int                 count     = 0;
    auto                predicate = [&count](const size_t& val)
    {
        const bool result = (val <= (size_t{1} << MaxPow2));
        count += (result ? 1 : 0);
        return result;
    };
    std::copy_if(test_sizes.begin(), test_sizes.end(), sizes.begin(), predicate);
    std::copy_if(random_sizes.begin(), random_sizes.end(), sizes.begin() + count, predicate);
    sizes.resize(count);

    std::sort(sizes.begin(), sizes.end());
    return sizes;
}

/// \brief Computes the closest multiple of \p divisor to a certain \p ref.
/// \param ref Number to be rounded up.
/// \param divisor Number which closest multiple to \p ref we are looking for.
inline size_t closest_greater_multiple(const size_t ref, const size_t divisor)
{
    if(!divisor)
    {
        return ref;
    }
    const size_t remainder = ref % divisor;
    size_t       distance  = remainder ? divisor - remainder : 0;
    return ref + distance;
}

template<class T>
std::vector<size_t> get_block_size_multiples(T seed_value, const unsigned int block_size)
{
    std::vector<size_t> sizes = get_sizes(seed_value);
    std::transform(sizes.begin(),
                   sizes.end(),
                   sizes.begin(),
                   [block_size](size_t size)
                   { return closest_greater_multiple(size, block_size); });
    std::set<size_t> unique_sizes(sizes.begin(), sizes.end());
    return std::vector<size_t>(unique_sizes.begin(), unique_sizes.end());
}

} // namespace test_utils

#endif //ROCPRIM_TEST_UTILS_DATA_GENERATION_HPP

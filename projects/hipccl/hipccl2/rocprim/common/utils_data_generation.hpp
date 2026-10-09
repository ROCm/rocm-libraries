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

#ifndef COMMON_UTILS_DATA_GENERATION_HPP_
#define COMMON_UTILS_DATA_GENERATION_HPP_

#include <rocprim/type_traits.hpp>
#include <rocprim/types.hpp>

#include <iterator>
#include <random>
#include <stdint.h>
#include <type_traits>

namespace common
{

// uniform_int_distribution is undefined for anything other than:
// short, int, long, long long, rocprim::int128_t, unsigned short, unsigned int, unsigned long, unsigned long long, or rocprim::uint128_t
template<typename T>
struct is_valid_for_int_distribution
    : std::integral_constant<
          bool,
          std::is_same<short, T>::value || std::is_same<unsigned short, T>::value
              || std::is_same<int, T>::value || std::is_same<unsigned int, T>::value
              || std::is_same<long, T>::value || std::is_same<unsigned long, T>::value
              || std::is_same<long long, T>::value || std::is_same<unsigned long long, T>::value
              || std::is_same<rocprim::int128_t, T>::value
              || std::is_same<rocprim::uint128_t, T>::value>
{};

// uniform_int_distribution is defined for supporting rocprim::int128_t and rocprim::uint128_t
template<typename IntType, typename Enable = void>
class uniform_int_distribution
{
public:
    typedef IntType result_type;

    uniform_int_distribution() : uniform_int_distribution(0) {}

    explicit uniform_int_distribution(IntType _a,
                                      IntType _b = rocprim::numeric_limits<IntType>::max())
        : lower_bound{_a}, upper_bound{_b}
    {}

    void reset() {}

    result_type a() const
    {
        return lower_bound;
    }

    result_type b() const
    {
        return upper_bound;
    }

    result_type min() const
    {
        return a();
    }

    result_type max() const
    {
        return b();
    }

    template<typename Generator>
    result_type operator()(Generator& urng)
    {
        rocprim::uint128_t range  = upper_bound - lower_bound + 1;
        auto               offset = helper(urng, range);
        return offset + lower_bound;
    }

    friend bool operator==(const uniform_int_distribution& d1, const uniform_int_distribution& d2)
    {
        return d1.lower_bound == d2.lower_bound && d1.upper_bound == d2.upper_bound;
    }

    friend bool operator!=(const uniform_int_distribution& d1, const uniform_int_distribution& d2)
    {
        return !(d1 == d2);
    }

    // third constructor, param(), operator<< and operator>> are not defined

private:
    // Java approach in the reference below.
    // Returns an unbiased random number from urng downscaled to [0, range)
    template<typename Generator>
    static rocprim::uint128_t helper(Generator& urng, const rocprim::uint128_t& range)
    {
        // reference: Fast Random Integer Geeneration in an Interval
        // ACM Transactions on Modeling and Computer Simulation 29 (1), 2019
        // https://arxiv.org/abs/1805.10941
        static std::uniform_int_distribution<uint64_t> dists[2];
        auto random_number = rocprim::uint128_t{dists[0](urng)} << 64 | dists[1](urng);
        if(!range)
        {
            return random_number;
        }
        auto result    = random_number % range;
        auto threshold = rocprim::numeric_limits<rocprim::uint128_t>::max() - range + 1;
        while(random_number - result > threshold)
        {
            random_number = rocprim::uint128_t{dists[0](urng)} << 64 | dists[1](urng);
            result        = random_number % range;
        }
        return result;
    }

    IntType lower_bound;
    IntType upper_bound;
};

template<typename IntType>
class uniform_int_distribution<
    IntType,
    std::enable_if_t<(!(std::is_same<rocprim::int128_t, IntType>::value
                        || std::is_same<rocprim::uint128_t, IntType>::value))>>
    : public std::uniform_int_distribution<IntType>
{
public:
    using std::uniform_int_distribution<IntType>::uniform_int_distribution;
};

template<typename T, typename enable = void>
struct generate_limits
{
    static inline T min()
    {
        return rocprim::numeric_limits<T>::min();
    }
    static inline T max()
    {
        return rocprim::numeric_limits<T>::max();
    }
};

template<typename T>
struct generate_limits<
    T,
    std::enable_if_t<rocprim::traits::get<T>().is_build_in() && rocprim::is_integral<T>::value>>
{
    static inline T min()
    {
        return rocprim::numeric_limits<T>::min();
    }
    static inline T max()
    {
        return rocprim::numeric_limits<T>::max();
    }
};

template<typename T>
struct generate_limits<T,
                       std::enable_if_t<rocprim::traits::get<T>().is_build_in()
                                        && rocprim::is_floating_point<T>::value>>
{
    static inline T min()
    {
        return T(-1000);
    }
    static inline T max()
    {
        return T(1000);
    }
};

template<typename Iterator>
using it_value_t = typename std::iterator_traits<Iterator>::value_type;

} // namespace common

#endif // COMMON_UTILS_DATA_GENERATION_HPP_

// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DETAIL_VARIOUS_HPP_
#define ROCPRIM_DETAIL_VARIOUS_HPP_

#include <type_traits>

#include "config.hpp"
#include "../types.hpp"

// TODO: Refactor when it gets crowded

BEGIN_ROCPRIM_NAMESPACE
namespace detail
{

struct empty_storage_type
{

};

template<class T>
constexpr bool is_power_of_two(const T x)
{
    static_assert(std::is_integral<T>::value, "T must be integer type");
    return (x > 0) && ((x & (x - 1)) == 0);
}

template<class T>
constexpr T next_power_of_two(const T x, const T acc = 1)
{
    static_assert(std::is_unsigned<T>::value, "T must be unsigned type");
    return acc >= x ? acc : next_power_of_two(x, 2 * acc);
}

template<class T>
inline constexpr
typename std::enable_if<std::is_integral<T>::value, T>::type
ceiling_div(T a, T b) [[hc]] [[cpu]]
{
    return (a + b - 1) / b;
}

// Select the minimal warp size for block of size block_size, it's
// useful for blocks smaller than maximal warp size.
template<class T>
constexpr T get_min_warp_size(const T block_size, const T max_warp_size)
{
    static_assert(std::is_unsigned<T>::value, "T must be unsigned type");
    return block_size >= max_warp_size ? max_warp_size : next_power_of_two(block_size);
}

template<unsigned int WarpSize>
struct is_warpsize_shuffleable {
    static const bool value = detail::is_power_of_two(WarpSize);
};

// Selects an appropriate vector_type based on the input T and size N.
// The byte size is calculated and used to select an appropriate vector_type.
template<class T, unsigned int N>
struct match_vector_type
{
    static constexpr unsigned int size = sizeof(T) * N;
    using vector_base_type =
        typename std::conditional<
            sizeof(T) >= 4,
            int,
            typename std::conditional<
                sizeof(T) >= 2,
                short,
                char
            >::type
        >::type;

    using vector_4 = typename make_vector_type<vector_base_type, 4>::type;
    using vector_2 = typename make_vector_type<vector_base_type, 2>::type;
    using vector_1 = typename make_vector_type<vector_base_type, 1>::type;

    using type =
        typename std::conditional<
            size % sizeof(vector_4) == 0,
            vector_4,
            typename std::conditional<
                size % sizeof(vector_2) == 0,
                vector_2,
                vector_1
            >::type
        >::type;
};

// Checks if Items is odd and ensures that size of T is smaller than vector_type.
template<class T, unsigned int Items>
inline constexpr bool is_vectorizable() [[hc]] [[cpu]]
{
    return (Items % 2 == 0) &&
           (sizeof(T) < sizeof(typename match_vector_type<T, Items>::type));
}

// Returns the number of LDS (local data share) banks.
constexpr unsigned int get_lds_banks_no()
{
    // Currently all devices supported by ROCm have 32 banks (4 bytes each)
    return 32;
}

} // end namespace detail
END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DETAIL_VARIOUS_HPP_

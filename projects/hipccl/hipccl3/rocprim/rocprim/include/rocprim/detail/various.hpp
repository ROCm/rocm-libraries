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

#include "../config.hpp"
#include "../types.hpp"
#include "../type_traits.hpp"

// TODO: Refactor when it gets crowded

BEGIN_ROCPRIM_NAMESPACE
namespace detail
{

struct empty_storage_type
{

};

template<class T>
ROCPRIM_HOST_DEVICE inline
constexpr bool is_power_of_two(const T x)
{
    static_assert(::rocprim::is_integral<T>::value, "T must be integer type");
    return (x > 0) && ((x & (x - 1)) == 0);
}

template<class T>
ROCPRIM_HOST_DEVICE inline
constexpr T next_power_of_two(const T x, const T acc = 1)
{
    static_assert(::rocprim::is_unsigned<T>::value, "T must be unsigned type");
    return acc >= x ? acc : next_power_of_two(x, 2 * acc);
}

template<class T>
ROCPRIM_HOST_DEVICE inline
constexpr auto ceiling_div(T a, T b)
    -> typename std::enable_if<::rocprim::is_integral<T>::value, T>::type
{
    return (a + b - 1) / b;
}

ROCPRIM_HOST_DEVICE inline
size_t align_size(size_t size, size_t alignment = 256)
{
    return ceiling_div(size, alignment) * alignment;
}

// Select the minimal warp size for block of size block_size, it's
// useful for blocks smaller than maximal warp size.
template<class T>
ROCPRIM_HOST_DEVICE inline
constexpr T get_min_warp_size(const T block_size, const T max_warp_size)
{
    static_assert(::rocprim::is_unsigned<T>::value, "T must be unsigned type");
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
ROCPRIM_HOST_DEVICE
constexpr bool is_vectorizable()
{
    return (Items % 2 == 0) &&
           (sizeof(T) < sizeof(typename match_vector_type<T, Items>::type));
}

// Returns the number of LDS (local data share) banks.
ROCPRIM_HOST_DEVICE
constexpr unsigned int get_lds_banks_no()
{
    // Currently all devices supported by ROCm have 32 banks (4 bytes each)
    return 32;
}

// Finds biggest fundamental type for type T that sizeof(T) is
// a multiple of that type's size.
template<class T>
struct match_fundamental_type
{
    using type =
        typename std::conditional<
            sizeof(T)%8 == 0,
            unsigned long long,
            typename std::conditional<
                sizeof(T)%4 == 0,
                unsigned int,
                typename std::conditional<
                    sizeof(T)%2 == 0,
                    unsigned short,
                    unsigned char
                >::type
            >::type
        >::type;
};

template<class T>
ROCPRIM_DEVICE inline
auto store_volatile(T * output, T value)
    -> typename std::enable_if<std::is_fundamental<T>::value>::type
{
    *const_cast<volatile T*>(output) = value;
}

template<class T>
ROCPRIM_DEVICE inline
auto store_volatile(T * output, T value)
    -> typename std::enable_if<!std::is_fundamental<T>::value>::type
{
    using fundamental_type = typename match_fundamental_type<T>::type;
    constexpr unsigned int n = sizeof(T) / sizeof(fundamental_type);

    auto input_ptr = reinterpret_cast<volatile fundamental_type*>(&value);
    auto output_ptr = reinterpret_cast<volatile fundamental_type*>(output);

    #pragma unroll
    for(unsigned int i = 0; i < n; i++)
    {
        output_ptr[i] = input_ptr[i];
    }
}

template<class T>
ROCPRIM_DEVICE inline
auto load_volatile(T * input)
    -> typename std::enable_if<std::is_fundamental<T>::value, T>::type
{
    T retval = *const_cast<volatile T*>(input);
    return retval;
}

template<class T>
ROCPRIM_DEVICE inline
auto load_volatile(T * input)
    -> typename std::enable_if<!std::is_fundamental<T>::value, T>::type
{
    using fundamental_type = typename match_fundamental_type<T>::type;
    constexpr unsigned int n = sizeof(T) / sizeof(fundamental_type);

    T retval;
    auto output_ptr = reinterpret_cast<volatile fundamental_type*>(&retval);
    auto input_ptr = reinterpret_cast<volatile fundamental_type*>(input);

    #pragma unroll
    for(unsigned int i = 0; i < n; i++)
    {
        output_ptr[i] = input_ptr[i];
    }
    return retval;
}

// A storage-backing wrapper that allows types with non-trivial constructors to be aliased in unions
template <typename T>
struct raw_storage
{
    // Biggest memory-access word that T is a whole multiple of and is not larger than the alignment of T
    typedef typename detail::match_fundamental_type<T>::type device_word;

    // Backing storage
    device_word storage[sizeof(T) / sizeof(device_word)];

    // Alias
    ROCPRIM_HOST_DEVICE T& get()
    {
        return reinterpret_cast<T&>(*this);
    }
};

// Checks if two iterators have the same type and value
template<class Iterator1, class Iterator2>
inline
bool are_iterators_equal(Iterator1, Iterator2)
{
    return false;
}

template<class Iterator>
inline
bool are_iterators_equal(Iterator iter1, Iterator iter2)
{
    return iter1 == iter2;
}

} // end namespace detail
END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DETAIL_VARIOUS_HPP_

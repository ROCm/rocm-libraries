// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

/// \file cpp_utils.hpp
/// Utility templates to provide extended functionality to standard library types.

#ifndef ROCRAND_RNG_UTILS_CPP_UTILS_HPP_
#define ROCRAND_RNG_UTILS_CPP_UTILS_HPP_

#include "unreachable.hpp"

#include <hip/hip_runtime_api.h>

#include <array>
#include <tuple>
#include <type_traits>

#include <cassert>
#include <cstddef>

namespace rocrand_impl::cpp_utils
{

/// \brief Invoke a function on each element of a \c std::tuple separately.
/// \tparam Functor Type of the function.
/// \tparam Tuple Type of the \c std::tuple.
/// \param fun Functor to invoke.
/// \param tuple Tuple whose every element is passed to \c fun.
template<class Functor, class Tuple>
constexpr void visit_tuple(Functor&& fun, Tuple&& tuple)
{
    std::apply([&fun](auto&&... args) { ((fun(args)), ...); }, tuple);
}

template<class T>
struct array_size
{
    // always false
    static_assert(sizeof(T) == 0, "T must be an std::array");
};

template<class T, std::size_t N>
struct array_size<std::array<T, N>>
{
    static constexpr inline std::size_t value = N;
};

/// \brief Returns the element count (size) of an \ref std::array.
/// \tparam T Type of the \ref std::array.
template<class T>
constexpr std::size_t array_size_v = array_size<T>::value;

/// \brief Generates all combinations of values of the input arrays.
/// \tparam T Element type of the input arrays.
/// \tparam ...Ns Sizes of the input arrays.
/// \param ...inputs Input array args.
/// \return An array of arrays. Each array in the returned array contains
/// one unique combination of the values from the input arrays.
template<class T, std::size_t... Ns>
constexpr auto numeric_combinations(const std::array<T, Ns>... inputs)
{
    constexpr std::size_t dim = sizeof...(Ns);
    static_assert(dim >= 1, "There must be at least one argument.");
    constexpr std::size_t                   product = (Ns * ...);
    std::array<std::array<T, dim>, product> ret{};

    // keeps track of the column being written
    std::size_t col = 0;
    // keeps track of how many rows we have to write the same value in the current column
    // to generate all combinations.
    std::size_t stride = 1;
    // Invoke a lambda on all inputs, building the output column-by-column
    ((
         [&]
         {
             // inputs.size() should also work, but it doesn't with NVCC
             constexpr std::size_t N = array_size_v<std::decay_t<decltype(inputs)>>;
             static_assert(N > 0, "Each array must have at least one element.");
             for(std::size_t i = 0; i < product; ++i)
             {
                 ret[i][col] = inputs[i / stride % N];
             }
             stride *= N;
             ++col;
         }()),
     ...);

    return ret;
}

/// \brief Calculates greatest common divisor.
__host__ __device__ constexpr unsigned int gcd(const unsigned int a, const unsigned int b)
{
    if(a == 0)
        return b;
    if(b == 0)
        return a;
    return gcd(b, a % b);
}

/// \brief Calculates least common multiple
__host__ __device__ constexpr unsigned int lcm(const unsigned int a, const unsigned int b)
{
    if(a == 0 || b == 0)
        return 0;
    return a / gcd(a, b) * b;
}

/// \brief Whether the type argument is a HIP vector type, e.g. `float4`.
template<class V>
constexpr bool is_vector_type_v = false;

template<class V>
struct vector_size
{
    // Always false
    static_assert(sizeof(V) == 0, "vector_size must be used with a vector type");
};

/// \brief The element count of the vector type, e.g. 2 for `float2`.
template<class V>
constexpr size_t vector_size_v = vector_size<V>::value;

template<class V>
struct vector_element
{
    // Always false
    static_assert(sizeof(V) == 0, "vector_element must be used with a vector type");
};

/// \brief The element type of the vector type, e.g. `float` for `float2`.
template<class V>
using vector_element_t = typename vector_element<V>::type;

#define INSTANTIATE_VECTOR_TYPE_TRAITS(V, T, N)       \
    template<>                                        \
    constexpr inline bool is_vector_type_v<V> = true; \
    template<>                                        \
    struct vector_size<V>                             \
    {                                                 \
        static constexpr inline size_t value = N;     \
    };                                                \
    template<>                                        \
    struct vector_element<V>                          \
    {                                                 \
        using type = T;                               \
    };

#define INSTANTIATE_VECTOR_TYPE_TRAITS_FOR_ELEMENT(base, T) \
    INSTANTIATE_VECTOR_TYPE_TRAITS(base##2, T, 2)           \
    INSTANTIATE_VECTOR_TYPE_TRAITS(base##4, T, 4)

INSTANTIATE_VECTOR_TYPE_TRAITS_FOR_ELEMENT(uchar, unsigned char)
INSTANTIATE_VECTOR_TYPE_TRAITS_FOR_ELEMENT(ushort, unsigned short)
INSTANTIATE_VECTOR_TYPE_TRAITS_FOR_ELEMENT(uint, unsigned int)
INSTANTIATE_VECTOR_TYPE_TRAITS_FOR_ELEMENT(ulong, unsigned long)
INSTANTIATE_VECTOR_TYPE_TRAITS_FOR_ELEMENT(ulonglong, unsigned long long int)

INSTANTIATE_VECTOR_TYPE_TRAITS_FOR_ELEMENT(char, char)
INSTANTIATE_VECTOR_TYPE_TRAITS_FOR_ELEMENT(short, short)
INSTANTIATE_VECTOR_TYPE_TRAITS_FOR_ELEMENT(int, int)
INSTANTIATE_VECTOR_TYPE_TRAITS_FOR_ELEMENT(long, long)
INSTANTIATE_VECTOR_TYPE_TRAITS_FOR_ELEMENT(longlong, long long int)

INSTANTIATE_VECTOR_TYPE_TRAITS_FOR_ELEMENT(float, float)
INSTANTIATE_VECTOR_TYPE_TRAITS_FOR_ELEMENT(double, double)

#undef INSTANTIATE_VECTOR_TYPE_TRAITS

/// \brief Wraps an instance of a vector type and provides indexing operators.
/// \tparam V The wrapped vector type.
template<class V>
struct vec_wrapper
{
    static_assert(is_vector_type_v<V>, "vec_wrapper can only be used with vector types");

    __host__ __device__ explicit vec_wrapper(V vec) : m_vec(vec) {}

    template<class U = V, std::enable_if_t<vector_size_v<U> == 2, int> = 0>
    __host__ __device__ auto& operator[](int idx)
    {
        switch(idx)
        {
            case 0: return m_vec.x;
            case 1: return m_vec.y;
        }
        ROCRAND_UNREACHABLE("Out-of-bounds indexing of vector");
    }

    template<class U = V, std::enable_if_t<vector_size_v<U> == 2, int> = 0>
    __host__ __device__ const auto& operator[](int idx) const
    {
        switch(idx)
        {
            case 0: return m_vec.x;
            case 1: return m_vec.y;
        }
        ROCRAND_UNREACHABLE("Out-of-bounds indexing of vector");
    }

    template<class U = V, std::enable_if_t<vector_size_v<U> == 4, int> = 0>
    __host__ __device__ auto& operator[](int idx)
    {
        switch(idx)
        {
            case 0: return m_vec.x;
            case 1: return m_vec.y;
            case 2: return m_vec.z;
            case 3: return m_vec.w;
        }
        ROCRAND_UNREACHABLE("Out-of-bounds indexing of vector");
    }

    template<class U = V, std::enable_if_t<vector_size_v<U> == 4, int> = 0>
    __host__ __device__ const auto& operator[](int idx) const
    {
        switch(idx)
        {
            case 0: return m_vec.x;
            case 1: return m_vec.y;
            case 2: return m_vec.z;
            case 3: return m_vec.w;
        }
        ROCRAND_UNREACHABLE("Out-of-bounds indexing of vector");
    }

    V m_vec;
};

template<class V>
__host__ __device__ vec_wrapper(V) -> vec_wrapper<V>;

/// \brief Returns the maximum of its arguments.
/// \note This function must be the choice in `__host__ __device__` and preferably on
/// `__device__` functions too. In host code, both this and `std::max` is allowed.
///
/// The reason for that is that the platform-provided  `__clang_hip_math.h` header provides
/// different overload sets for `::min` and `::max` between device and host code. That implementation
/// can result in unwanted implicit conversions to `int` in host code.
template<class T>
__host__ __device__
constexpr T max(const T& a, const T& b)
{
    return a < b ? b : a;
}

/// \brief Returns the minimum of its arguments.
/// \note This function must be the choice in `__host__ __device__` and preferably on
/// `__device__` functions too. In host code, both this and `std::min` is allowed.
///
/// The reason for that is that the platform-provided  `__clang_hip_math.h` header provides
/// different overload sets for `::min` and `::max` between device and host code. That implementation
/// can result in unwanted implicit conversions to `int` in host code.
template<class T>
__host__ __device__
constexpr T min(const T& a, const T& b)
{
    return a < b ? a : b;
}

} // end namespace rocrand_impl::cpp_utils

#endif // ROCRAND_RNG_CPP_UTILS_HPP_

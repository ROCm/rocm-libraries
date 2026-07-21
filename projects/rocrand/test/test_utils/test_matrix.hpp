// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef TEST_TEST_UTILS_TEST_MATRIX_HPP_
#define TEST_TEST_UTILS_TEST_MATRIX_HPP_

#include <gtest/gtest.h>

#include <tuple>

namespace test_utils
{

template<typename... T>
struct append;
template<typename... Ts, typename... Us>
struct append<std::tuple<Ts...>, std::tuple<Us...>>
{
    using type = std::tuple<Ts..., Us...>;
};

template<typename... T>
using append_t = typename append<T...>::type;

template<typename T, typename U>
struct cartesian_product;

// Base case - an empty tuple combined with anything results in an empty tuple
template<typename... Us>
struct cartesian_product<std::tuple<>, std::tuple<Us...>>
{
    using type = std::tuple<>;
};

// Alias to differentiate the tuple that is an element of the resulting
// test matrix from the tuples that form the lists of input types
template<typename GeneratorType, typename DistributionType>
using rocrand_test_tuple_t = std::tuple<GeneratorType, DistributionType>;

template<typename T, typename... Ts, typename... Us>
struct cartesian_product<std::tuple<T, Ts...>, std::tuple<Us...>>
{
    // Take first type of first tuple and combine it with the complete list of types of the second tuple
    using head_combination_tuple = std::tuple<rocrand_test_tuple_t<T, Us>...>;

    // Repeat for remainder of first tuple with complete second tuple
    using tail_combination = typename cartesian_product<std::tuple<Ts...>, std::tuple<Us...>>::type;

    using type = append_t<head_combination_tuple, tail_combination>;
};

template<typename... Ts>
struct test_matrix_impl;

template<typename... Ts>
struct test_matrix_impl<std::tuple<Ts...>>
{
    using test_types = testing::Types<Ts...>;
};

template<typename... Ts>
struct test_matrix;

// creates cartesian product of two tuples of types to be tested
template<typename... Ts, typename... Us>
struct test_matrix<std::tuple<Ts...>, std::tuple<Us...>>
{
    using test_types = typename test_matrix_impl<
        typename cartesian_product<std::tuple<Ts...>, std::tuple<Us...>>::type>::test_types;
};

} // namespace test_utils

#endif // TEST_TEST_UTILS_TEST_MATRIX_HPP_

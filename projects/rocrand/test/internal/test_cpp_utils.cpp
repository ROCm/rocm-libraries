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

#include "rng/utils/cpp_utils.hpp"
#include "rng/utils/threedim_iterator.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <stdint.h>

using namespace rocrand_impl;

TEST(rocrand_cpp_utils_tests, visit_tuple)
{
    std::tuple               t{1, std::size_t(2), (unsigned short)(3)};
    std::vector<std::string> results;

    cpp_utils::visit_tuple([&](auto&& val) { results.push_back(std::to_string(val)); }, t);

    ASSERT_EQ("1", results.at(0));
    ASSERT_EQ("2", results.at(1));
    ASSERT_EQ("3", results.at(2));
}

TEST(rocrand_cpp_utils_tests, numeric_combinations)
{
    constexpr std::array A{1, 2, 3, 4};
    constexpr std::array B{10, 20, 30};
    constexpr std::array C{100, 200};

    constexpr auto combinations = cpp_utils::numeric_combinations(A, B, C);

    ASSERT_EQ(combinations.size(), A.size() * B.size() * C.size());

    const std::set<std::array<int, 3>> combination_set(combinations.begin(), combinations.end());
    ASSERT_EQ(combinations.size(), combination_set.size()) << "Not all items are unique";

    for(auto [a, b, c] : combinations)
    {
        ASSERT_NE(std::find(A.begin(), A.end(), a), A.end()) << "Element not found in A";
        ASSERT_NE(std::find(B.begin(), B.end(), b), B.end()) << "Element not found in B";
        ASSERT_NE(std::find(C.begin(), C.end(), c), C.end()) << "Element not found in C";
    }
}

TEST(rocrand_cpp_utils_tests, gcd)
{
    using namespace cpp_utils;

    ASSERT_EQ(gcd(127, 3), 1);
    ASSERT_EQ(gcd(2048, 512), 512);
    ASSERT_EQ(gcd(12, 18), 6);
}

TEST(rocrand_cpp_utils_tests, lcm)
{
    using namespace cpp_utils;
    ASSERT_EQ(lcm(0, 0), 0);
    ASSERT_EQ(lcm(1, 0), 0);
    ASSERT_EQ(lcm(0, 1), 0);

    ASSERT_EQ(lcm(127, 3), 127 * 3);
    ASSERT_EQ(lcm(2048, 512), 2048);
    ASSERT_EQ(lcm(12, 18), 36);
}

TEST(rocrand_cpp_utils_tests, vec_wrapper)
{
    using namespace cpp_utils;
    const int4  f{1, 2, 3, 4};
    vec_wrapper wrapper(f);

    ASSERT_EQ(1, wrapper[0]);
    ASSERT_EQ(2, wrapper[1]);
    ASSERT_EQ(3, wrapper[2]);
    ASSERT_EQ(4, wrapper[3]);

    wrapper[2] = 100;
    ASSERT_EQ(100, wrapper[2]);
    ASSERT_EQ(3, f.z);
}

static bool operator==(const dim3& lhs, const dim3& rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

TEST(rocrand_cpp_utils_tests, threedim_iterator)
{
    using namespace cpp_utils;

    const dim3        dim(100, 256, 4);
    threedim_iterator it(dim);
    ASSERT_EQ(it, threedim_iterator::begin(dim));
    ASSERT_NE(it, threedim_iterator::end(dim));
    {
        const dim3 d = *it;
        const dim3 expected{0, 0, 0};
        ASSERT_EQ(d, expected);
    }

    std::advance(it, 125);
    {
        const dim3 d = *it;
        const dim3 expected{25, 1, 0};
        ASSERT_EQ(d, expected);
    }

    std::advance(it, 68322);
    {
        const dim3 d = *it;
        const dim3 expected{47, 172, 2};
        ASSERT_EQ(d, expected);
    }

    std::advance(it, 33953);
    ASSERT_EQ(it, threedim_iterator::end(dim));
}

TEST(rocrand_cpp_utils_tests, threedim_iterator_list)
{
    using namespace cpp_utils;

    const dim3        dim{3, 2, 4};
    std::vector<dim3> results;
    std::copy(threedim_iterator::begin(dim),
              threedim_iterator::end(dim),
              std::back_inserter(results));

    ASSERT_EQ(dim.x * dim.y * dim.z, results.size());
    size_t index = 0;
    for(uint32_t z = 0; z < dim.z; ++z)
    {
        for(uint32_t y = 0; y < dim.y; ++y)
        {
            for(uint32_t x = 0; x < dim.x; ++x, ++index)
            {
                const dim3 expected{x, y, z};
                ASSERT_EQ(expected, results[index]);
            }
        }
    }
}

// must be in global namespace
static bool operator<(const dim3& /*lhs*/, const dim3& /*rhs*/)
{
    // not relevant for the test below
    return true;
}

TEST(rocrand_cpp_utils_tests, threedim_iterator_check_stl)
{
    // std::is_heap requires a named concept LegacyRandomAccessIterator for the iterator
    // This is a primitive test to check whether threedim_iterator fulfills this requirement
    //
    // This is not full coverage of the required members of the aforementioned requirement,
    // since there is no guarantee that the algorithm would call every required member function
    //
    // A better solution will be checking against the std::random_access_iterator_concept in C++20
    (void)std::is_heap(cpp_utils::threedim_iterator(), cpp_utils::threedim_iterator());
}

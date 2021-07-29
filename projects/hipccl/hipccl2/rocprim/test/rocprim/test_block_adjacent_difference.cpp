// MIT License
//
// Copyright (c) 2017-2021 Advanced Micro Devices, Inc. All rights reserved.
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

#include "common_test_header.hpp"

// required rocprim headers
#include <rocprim/config.hpp>
#include <rocprim/block/block_adjacent_difference.hpp>
#include <rocprim/block/block_load.hpp>
#include <rocprim/block/block_store.hpp>

// required test headers
#include "test_utils_types.hpp"

// kernel definitions
#include "test_block_adjacent_difference.kernels.hpp"

TEST(RocprimBlockAdjacentDifference, Traits)
{
    ASSERT_FALSE((rocprim::detail::with_b_index_arg<int, rocprim::less<int>>::value));
    ASSERT_FALSE((rocprim::detail::with_b_index_arg<int, custom_flag_op2>::value));
    ASSERT_TRUE((rocprim::detail::with_b_index_arg<int, custom_flag_op1<int>>::value));

    auto f1 = [](const int& a, const int& b, unsigned int b_index) { return (a == b) || (b_index % 10 == 0); };
    auto f2 = [](const int& a, const int& b) { return (a == b); };
    ASSERT_TRUE((rocprim::detail::with_b_index_arg<int, decltype(f1)>::value));
    ASSERT_FALSE((rocprim::detail::with_b_index_arg<int, decltype(f2)>::value));

    auto f3 = [](int a, int b, int b_index) { return (a == b) || (b_index % 10 == 0); };
    auto f4 = [](const int a, const int b) { return (a == b); };
    ASSERT_TRUE((rocprim::detail::with_b_index_arg<int, decltype(f3)>::value));
    ASSERT_FALSE((rocprim::detail::with_b_index_arg<int, decltype(f4)>::value));
}

// Start stamping out tests
struct RocprimBlockAdjacentDifference;

struct Integral;
#define suite_name RocprimBlockAdjacentDifference
#define warp_params BlockDiscParamsIntegral
#define name_suffix Integral

#include "test_block_adjacent_difference.hpp"

#undef suite_name
#undef warp_params
#undef name_suffix

struct Floating;
#define suite_name RocprimBlockAdjacentDifference
#define warp_params BlockDiscParamsFloating
#define name_suffix Floating

#include "test_block_adjacent_difference.hpp"

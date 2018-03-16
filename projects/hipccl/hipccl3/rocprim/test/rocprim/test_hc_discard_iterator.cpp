// MIT License
//
// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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

#include <iostream>
#include <vector>
#include <algorithm>
#include <type_traits>

// Google Test
#include <gtest/gtest.h>
// HC API
#include <hcc/hc.hpp>
// rocPRIM API
#include <rocprim/rocprim.hpp>

#include "test_utils.hpp"

TEST(RocprimDiscardIteratorTests, Equal)
{
    using Iterator = typename rocprim::discard_iterator;

    Iterator x(test_utils::get_random_value<size_t>(0, 200));
    Iterator y = x;
    ASSERT_EQ(x, y);

    x += 100;
    for(size_t i = 0; i < 100; i++)
    {
        y++;
    }
    ASSERT_EQ(x, y);

    y--;
    ASSERT_NE(x, y);
}

TEST(RocprimDiscardIteratorTests, Less)
{
    using Iterator = typename rocprim::discard_iterator;

    Iterator x(test_utils::get_random_value<size_t>(0, 200));
    Iterator y = x + 1;
    ASSERT_LT(x, y);

    x += 100;
    for(size_t i = 0; i < 100; i++)
    {
        y++;
    }
    ASSERT_LT(x, y);
}

TEST(RocprimDiscardIteratorTests, ReduceByKey)
{
    const bool debug_synchronous = false;

    hc::accelerator acc;
    hc::accelerator_view acc_view = acc.create_view();

    // host input
    std::vector<int> keys_input = {
        0, 0, 0, 1, 1, 2, 2, 0, 0, 0, 0
    };
    std::vector<int> values_input(keys_input.size(), 1);

    // expected output
    std::vector<int> aggregates_expected = { 3, 2, 2, 4 };

    // device input/output
    hc::array<int> d_keys_input(
        hc::extent<1>(keys_input.size()), keys_input.begin(), acc_view
    );
    hc::array<int> d_values_input(
        hc::extent<1>(values_input.size()), values_input.begin(), acc_view
    );
    hc::array<int> d_aggregates_output(4, acc_view);
    acc_view.wait();

    // Get temporary storage size
    size_t temporary_storage_bytes;
    rocprim::reduce_by_key(
        nullptr, temporary_storage_bytes,
        d_keys_input.accelerator_pointer(),
        d_values_input.accelerator_pointer(), values_input.size(),
        rocprim::make_discard_iterator(),
        d_aggregates_output.accelerator_pointer(),
        rocprim::make_discard_iterator(),
        rocprim::plus<int>(), rocprim::equal_to<int>(),
        acc_view, debug_synchronous
    );
    acc_view.wait();

    ASSERT_GT(temporary_storage_bytes, 0);

    hc::array<char> d_temporary_storage(temporary_storage_bytes, acc_view);
    acc_view.wait();

    rocprim::reduce_by_key(
        d_temporary_storage.accelerator_pointer(), temporary_storage_bytes,
        d_keys_input.accelerator_pointer(),
        d_values_input.accelerator_pointer(), values_input.size(),
        rocprim::make_discard_iterator(),
        d_aggregates_output.accelerator_pointer(),
        rocprim::make_discard_iterator(),
        rocprim::plus<int>(), rocprim::equal_to<int>(),
        acc_view, debug_synchronous
    );
    acc_view.wait();

    // Check if output values are as expected
    std::vector<int> aggregates_output = d_aggregates_output;
    for(size_t i = 0; i < aggregates_output.size(); i++)
    {
        ASSERT_EQ(aggregates_output[i], aggregates_expected[i]);
    }
}

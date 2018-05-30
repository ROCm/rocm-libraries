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

// Params for tests
template<class InputType>
struct RocprimArgIndexIteratorParams
{
    using input_type = InputType;
};

template<class Params>
class RocprimArgIndexIteratorTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
    const bool debug_synchronous = false;
};

typedef ::testing::Types<
    RocprimArgIndexIteratorParams<int>,
    RocprimArgIndexIteratorParams<unsigned int>,
    RocprimArgIndexIteratorParams<unsigned long>,
    RocprimArgIndexIteratorParams<float>
> RocprimArgIndexIteratorTestsParams;

TYPED_TEST_CASE(RocprimArgIndexIteratorTests, RocprimArgIndexIteratorTestsParams);

TYPED_TEST(RocprimArgIndexIteratorTests, Equal)
{
    using T = typename TestFixture::input_type;
    using Iterator = typename rocprim::arg_index_iterator<T*>;
    //using key_value = typename Iterator::value_type;

    std::vector<T> input = test_utils::get_random_data<T>(5, 1, 200);

    Iterator x(input.data());
    Iterator y = x;
    for(size_t i = 0; i < 5; i++)
    {
        ASSERT_EQ(x[i].key, i);
        ASSERT_EQ(x[i].value, input[i]);
    }
    ASSERT_EQ(x[2].value, input[2]);

    x += 2;
    for(size_t i = 0; i < 2; i++)
    {
        y++;
    }
    ASSERT_EQ(x, y);
}

struct arg_min
{
    template<
        class Key,
        class Value
    >
    ROCPRIM_HOST_DEVICE inline
    constexpr rocprim::key_value_pair<Key, Value>
    operator()(const rocprim::key_value_pair<Key, Value>& a,
               const rocprim::key_value_pair<Key, Value>& b) const
    {
        return ((b.value < a.value) || ((a.value == b.value) && (b.key < a.key))) ? b : a;
    }
};

TYPED_TEST(RocprimArgIndexIteratorTests, ReduceArgMinimum)
{
    using T = typename TestFixture::input_type;
    using Iterator = typename rocprim::arg_index_iterator<T*>;
    using key_value = typename Iterator::value_type;
    using difference_type = typename Iterator::difference_type;
    const bool debug_synchronous = false;

    hc::accelerator acc;
    hc::accelerator_view acc_view = acc.create_view();

    const size_t size = 1024;
    // Generate data
    std::vector<T> input = test_utils::get_random_data<T>(size, 1, 200);

    hc::array<T> d_input(hc::extent<1>(size), input.data(), acc_view);
    hc::array<key_value> d_output(1, acc_view);
    acc_view.wait();

    arg_min reduce_op;
    const key_value max(std::numeric_limits<difference_type>::max(), std::numeric_limits<T>::max());

    // Calculate expected results on host
    Iterator x(input.data());
    key_value expected = std::accumulate(x, x + size, max, reduce_op);

    Iterator d_iter(d_input.accelerator_pointer());

    // temp storage
    size_t temp_storage_size_bytes;
    // Get size of d_temp_storage
    rocprim::reduce(
        nullptr,
        temp_storage_size_bytes,
        d_iter,
        d_output.accelerator_pointer(),
        max,
        input.size(),
        reduce_op,
        acc_view,
        debug_synchronous
    );
    acc_view.wait();

    // temp_storage_size_bytes must be >0
    ASSERT_GT(temp_storage_size_bytes, 0);

    // allocate temporary storage
    hc::array<char> d_temp_storage(temp_storage_size_bytes, acc_view);
    acc_view.wait();

    // Run
    rocprim::reduce(
        d_temp_storage.accelerator_pointer(),
        temp_storage_size_bytes,
        d_iter,
        d_output.accelerator_pointer(),
        max,
        input.size(),
        reduce_op,
        acc_view,
        debug_synchronous
    );
    acc_view.wait();

    // Check if output values are as expected
    std::vector<key_value> output = d_output;
    auto diff = std::max<T>(std::abs(0.01f * expected.value), T(0.01f));
    if(std::is_integral<T>::value) diff = 0;
    ASSERT_EQ(output[0].key, expected.key);
    ASSERT_NEAR(output[0].value, expected.value, diff);
}

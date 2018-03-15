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

template<class T>
struct times_two
{
    T operator()(const T& value) const
    {
        return 2 * value;
    }
};

template<class T>
struct plus_ten
{
    T operator()(const T& value) const
    {
        return value + 10;
    }
};

// Params for tests
template<
    class InputType,
    class UnaryFunction = times_two<InputType>,
    class ValueType = InputType
>
struct RocprimTransformIteratorParams
{
    using input_type = InputType;
    using value_type = ValueType;
    using unary_function = UnaryFunction;
};

template<class Params>
class RocprimTransformIteratorTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
    using value_type = typename Params::value_type;
    using unary_function = typename Params::unary_function;
    const bool debug_synchronous = false;
};

typedef ::testing::Types<
    RocprimTransformIteratorParams<int, plus_ten<long>>,
    RocprimTransformIteratorParams<unsigned int>,
    RocprimTransformIteratorParams<unsigned long>,
    RocprimTransformIteratorParams<float, plus_ten<double>, double>
> RocprimTransformIteratorTestsParams;

TYPED_TEST_CASE(RocprimTransformIteratorTests, RocprimTransformIteratorTestsParams);

TYPED_TEST(RocprimTransformIteratorTests, Equal)
{
    using input_type = typename TestFixture::input_type;
    using value_type = typename TestFixture::value_type;
    using unary_function = typename TestFixture::unary_function;
    using iterator_type = typename rocprim::transform_iterator<
        input_type*, unary_function, value_type
    >;

    std::vector<input_type> input =
        test_utils::get_random_data<input_type>(5, 1, 200);
    unary_function transform;

    iterator_type x(input.data(), transform);
    iterator_type y = x;
    for(size_t i = 0; i < 5; i++)
    {
        if (std::is_integral<value_type>::value)
        {
            ASSERT_EQ(x[i], static_cast<value_type>(transform(input[i])));
        }
        else if(std::is_floating_point<value_type>::value)
        {
            auto expected = static_cast<value_type>(transform(input[i]));
            auto tolerance = std::max<value_type>(std::abs(0.1f * expected), value_type(0.01f));
            ASSERT_NEAR(x[i], expected, tolerance);
        }
    }

    x += 100;
    for(size_t i = 0; i < 100; i++)
    {
        y++;
    }
    ASSERT_EQ(x, y);
}

TYPED_TEST(RocprimTransformIteratorTests, TransformReduce)
{
    using input_type = typename TestFixture::input_type;
    using value_type = typename TestFixture::value_type;
    using unary_function = typename TestFixture::unary_function;
    using iterator_type = typename rocprim::transform_iterator<
        input_type*, unary_function, value_type
    >;

    hc::accelerator acc;
    hc::accelerator_view acc_view = acc.create_view();

    const size_t size = 1024;
    // Generate data
    std::vector<input_type> input = test_utils::get_random_data<input_type>(size, 1, 200);

    hc::array<input_type> d_input(hc::extent<1>(size), input.data(), acc_view);
    hc::array<value_type> d_output(1, acc_view);
    acc_view.wait();

    auto reduce_op = rocprim::plus<value_type>();
    unary_function transform;

    // Calculate expected results on host
    iterator_type x(input.data(), transform);
    value_type expected = std::accumulate(x, x + size, value_type(0), reduce_op);

    auto d_iter = iterator_type(d_input.accelerator_pointer(), transform);

    // temp storage
    size_t temp_storage_size_bytes;
    // Get size of d_temp_storage
    rocprim::reduce(
        nullptr,
        temp_storage_size_bytes,
        d_iter,
        d_output.accelerator_pointer(),
        value_type(0),
        input.size(),
        reduce_op,
        acc_view
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
        value_type(0),
        input.size(),
        reduce_op,
        acc_view,
        TestFixture::debug_synchronous
    );
    acc_view.wait();

    // Check if output values are as expected
    std::vector<value_type> output = d_output;
    if(std::is_integral<value_type>::value)
    {
        ASSERT_EQ(output[0], expected);
    }
    else if(std::is_floating_point<value_type>::value)
    {
        auto tolerance = std::max<value_type>(std::abs(0.1f * expected), value_type(0.01f));
        ASSERT_NEAR(output[0], expected, tolerance);
    }
}

TYPED_TEST(RocprimTransformIteratorTests, TransformReduceCountingIterator)
{
    using input_type = typename TestFixture::input_type;
    using unary_function = typename TestFixture::unary_function;
    using counting_iterator_type = rocprim::counting_iterator<input_type>;

    hc::accelerator acc;
    hc::accelerator_view acc_view = acc.create_view();

    const size_t size = 1024;
    auto reduce_op = rocprim::plus<input_type>();
    unary_function transform;
    counting_iterator_type citer(0);

    // output
    hc::array<input_type> d_output(1, acc_view);

    // Calculate expected results on host
    input_type expected = std::accumulate(
        rocprim::make_transform_iterator(citer, transform),
        rocprim::make_transform_iterator(citer + size, transform),
        input_type(0),
        reduce_op
    );

    // temp storage
    size_t temp_storage_size_bytes;
    // Get size of d_temp_storage
    rocprim::reduce(
        nullptr,
        temp_storage_size_bytes,
        rocprim::make_transform_iterator(citer, transform),
        d_output.accelerator_pointer(),
        input_type(0),
        size,
        reduce_op,
        acc_view
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
        rocprim::make_transform_iterator(citer, transform),
        d_output.accelerator_pointer(),
        input_type(0),
        size,
        reduce_op,
        acc_view,
        TestFixture::debug_synchronous
    );
    acc_view.wait();

    // Check if output values are as expected
    std::vector<input_type> output = d_output;
    if(std::is_integral<input_type>::value)
    {
        ASSERT_EQ(output[0], expected);
    }
    else if(std::is_floating_point<input_type>::value)
    {
        auto tolerance = std::max<input_type>(std::abs(0.1f * expected), input_type(0.01f));
        ASSERT_NEAR(output[0], expected, tolerance);
    }
}


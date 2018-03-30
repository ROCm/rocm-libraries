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

// Google Test
#include <gtest/gtest.h>
// HC API
#include <hcc/hc.hpp>
// rocPRIM API
#include <rocprim/rocprim.hpp>

#include "test_utils.hpp"

namespace rp = rocprim;

// Params for tests
template<
    class InputType,
    class OutputType = InputType
>
struct DeviceTransformParams
{
    using input_type = InputType;
    using output_type = OutputType;
};

// ---------------------------------------------------------
// Test for reduce ops taking single input value
// ---------------------------------------------------------

template<class Params>
class RocprimDeviceTransformTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
    using output_type = typename Params::output_type;
    const bool debug_synchronous = false;
};

typedef ::testing::Types<
    DeviceTransformParams<int, long>,
    DeviceTransformParams<unsigned char, float>
> RocprimDeviceTransformTestsParams;

std::vector<size_t> get_sizes()
{
    std::vector<size_t> sizes = {
        2, 32, 65, 378,
        1512, 3048, 4096,
        27845, (1 << 18) + 1111
    };
    const std::vector<size_t> random_sizes = test_utils::get_random_data<size_t>(2, 1, 16384);
    sizes.insert(sizes.end(), random_sizes.begin(), random_sizes.end());
    std::sort(sizes.begin(), sizes.end());
    return sizes;
}

TYPED_TEST_CASE(RocprimDeviceTransformTests, RocprimDeviceTransformTestsParams);

template<class T>
struct transform
{
    inline
    constexpr T operator()(const T& a) const [[hc]] [[cpu]]
    {
        return a + 5;
    }
};

TYPED_TEST(RocprimDeviceTransformTests, Transform)
{
    using T = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    hc::accelerator acc;
    hc::accelerator_view acc_view = acc.create_view();

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        std::vector<T> input = test_utils::get_random_data<T>(size, 1, 100);

        hc::array<T> d_input(hc::extent<1>(size), input.begin(), acc_view);
        hc::array<U> d_output(size, acc_view);
        acc_view.wait();

        // Calculate expected results on host
        std::vector<U> expected(input.size());
        std::transform(input.begin(), input.end(), expected.begin(), transform<U>());

        // Run
        rocprim::transform(
            d_input.accelerator_pointer(),
            d_output.accelerator_pointer(),
            input.size(),
            transform<U>(),
            acc_view,
            debug_synchronous
        );
        acc_view.wait();

        // Check if output values are as expected
        std::vector<U> output = d_output;
        for(size_t i = 0; i < output.size(); i++)
        {
            SCOPED_TRACE(testing::Message() << "where index = " << i);
            auto diff = std::max<U>(std::abs(0.01f * expected[i]), U(0.01f));
            if(std::is_integral<U>::value) diff = 0;
            ASSERT_NEAR(output[i], expected[i], diff);
        }
    }
}

template<class T1, class T2, class U>
struct binary_transform
{
    inline
    constexpr U operator()(const T1& a, const T2& b) const [[hc]] [[cpu]]
    {
        return a + b;
    }
};

TYPED_TEST(RocprimDeviceTransformTests, BinaryTransform)
{
    using T1 = typename TestFixture::input_type;
    using T2 = typename TestFixture::input_type;
    using U = typename TestFixture::output_type;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    hc::accelerator acc;
    hc::accelerator_view acc_view = acc.create_view();

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        // Generate data
        std::vector<T1> input1 = test_utils::get_random_data<T1>(size, 1, 100);
        std::vector<T2> input2 = test_utils::get_random_data<T2>(size, 1, 100);

        hc::array<T1> d_input1(hc::extent<1>(size), input1.begin(), acc_view);
        hc::array<T2> d_input2(hc::extent<1>(size), input2.begin(), acc_view);
        hc::array<U> d_output(size, acc_view);
        acc_view.wait();

        // Calculate expected results on host
        std::vector<U> expected(input1.size());
        std::transform(
            input1.begin(), input1.end(), input2.begin(),
            expected.begin(), binary_transform<T1, T2, U>()
        );

        // Run
        rocprim::transform(
            d_input1.accelerator_pointer(),
            d_input2.accelerator_pointer(),
            d_output.accelerator_pointer(),
            input1.size(),
            binary_transform<T1, T2, U>(),
            acc_view,
            debug_synchronous
        );
        acc_view.wait();

        // Check if output values are as expected
        std::vector<U> output = d_output;
        for(size_t i = 0; i < output.size(); i++)
        {
            SCOPED_TRACE(testing::Message() << "where index = " << i);
            auto diff = std::max<U>(std::abs(0.01f * expected[i]), U(0.01f));
            if(std::is_integral<U>::value) diff = 0;
            ASSERT_NEAR(output[i], expected[i], diff);
        }
    }
}

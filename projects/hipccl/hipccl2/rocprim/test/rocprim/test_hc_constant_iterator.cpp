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
struct RocprimConstantIteratorParams
{
    using input_type = InputType;
};

template<class Params>
class RocprimConstantIteratorTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
    const bool debug_synchronous = false;
};

typedef ::testing::Types<
    RocprimConstantIteratorParams<int>,
    RocprimConstantIteratorParams<unsigned int>,
    RocprimConstantIteratorParams<unsigned long>,
    RocprimConstantIteratorParams<float>
> RocprimConstantIteratorTestsParams;

TYPED_TEST_CASE(RocprimConstantIteratorTests, RocprimConstantIteratorTestsParams);

template<class T>
struct transform
{
    constexpr T operator()(const T& a) const [[hc]] [[cpu]]
    {
        return a + 10;
    }
};

TYPED_TEST(RocprimConstantIteratorTests, Equal)
{
    using T = typename TestFixture::input_type;
    using Iterator = typename rocprim::constant_iterator<T>;

    Iterator x(test_utils::get_random_value<T>(0, 200));
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

TYPED_TEST(RocprimConstantIteratorTests, Less)
{
    using T = typename TestFixture::input_type;
    using Iterator = typename rocprim::constant_iterator<T>;

    Iterator x(test_utils::get_random_value<T>(0, 200));
    Iterator y = x + 1;
    ASSERT_LT(x, y);

    x += 100;
    for(size_t i = 0; i < 100; i++)
    {
        y++;
    }
    ASSERT_LT(x, y);
}

TYPED_TEST(RocprimConstantIteratorTests, Transform)
{
    using T = typename TestFixture::input_type;
    using Iterator = typename rocprim::constant_iterator<T>;
    const bool debug_synchronous = TestFixture::debug_synchronous;

    const size_t size = 1024;
    hc::accelerator acc;
    hc::accelerator_view acc_view = acc.create_view();

    // Create constant_iterator<T> with random starting point
    const auto value = test_utils::get_random_value<T>(0, 200);
    Iterator input_begin(value);
    hc::array<T> d_output(size, acc_view);
    acc_view.wait();

    // Calculate expected results on host
    std::vector<T> expected(size, transform<T>()(value));

    // Run
    rocprim::transform(
        input_begin,
        d_output.accelerator_pointer(),
        size,
        transform<T>(),
        acc_view,
        debug_synchronous
    );
    acc_view.wait();

    // Check if output values are as expected
    std::vector<T> output = d_output;
    for(size_t i = 0; i < output.size(); i++)
    {
        SCOPED_TRACE(testing::Message() << "where index = " << i);
        if(std::is_integral<T>::value)
        {
            ASSERT_EQ(output[i], expected[i]);
        }
        else if(std::is_floating_point<T>::value)
        {
            auto tolerance = std::max<T>(std::abs(0.1f * expected[i]), T(0.01f));
            ASSERT_NEAR(output[i], expected[i], tolerance);
        }
    }
}

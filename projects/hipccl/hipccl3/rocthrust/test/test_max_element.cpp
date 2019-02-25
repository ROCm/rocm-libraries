// MIT License
//
// Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
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

// Google Test
#include <gtest/gtest.h>

// Thrust
#include <thrust/extrema.h>
#include <thrust/iterator/retag.h>

// HIP API
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#define HIP_CHECK(condition) ASSERT_EQ(condition, hipSuccess)
#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

#include "test_utils.hpp"

template<
class InputType
>
struct Params
{
    using input_type = InputType;
};

template<class Params>
class MaxElementTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
};

typedef ::testing::Types<
Params<thrust::host_vector<short>>,
Params<thrust::host_vector<int>>,
Params<thrust::host_vector<long long>>,
Params<thrust::host_vector<unsigned short>>,
Params<thrust::host_vector<unsigned int>>,
Params<thrust::host_vector<unsigned long long>>,
Params<thrust::host_vector<float>>,
Params<thrust::host_vector<double>>,
Params<thrust::device_vector<short>>,
Params<thrust::device_vector<int>>,
Params<thrust::device_vector<long long>>,
Params<thrust::device_vector<unsigned short>>,
Params<thrust::device_vector<unsigned int>>,
Params<thrust::device_vector<unsigned long long>>,
Params<thrust::device_vector<float>>,
Params<thrust::device_vector<double>>
> MaxElementTestsParams;

TYPED_TEST_CASE(MaxElementTests, MaxElementTestsParams);

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

TYPED_TEST(MaxElementTests, TestMaxElementSimple)
{
    using Vector = typename TestFixture::input_type;
    using T = typename Vector::value_type;

    Vector data(6);
    data[0] = 3;
    data[1] = 5;
    data[2] = 1;
    data[3] = 2;
    data[4] = 5;
    data[5] = 1;

    ASSERT_EQ( *thrust::max_element(data.begin(), data.end()), 5);
    ASSERT_EQ( thrust::max_element(data.begin(), data.end()) - data.begin(), 1);

    ASSERT_EQ( *thrust::max_element(data.begin(), data.end(), thrust::greater<T>()), 1);
    ASSERT_EQ( thrust::max_element(data.begin(), data.end(), thrust::greater<T>()) - data.begin(), 2);
}

TYPED_TEST(MaxElementTests, TestMaxElementWithTransform)
{
    using Vector = typename TestFixture::input_type;
    using T = typename Vector::value_type;
    
    // We cannot use unsigned types for this test case
    if (std::is_unsigned<T>::value)
        return;

    Vector data(6);
    data[0] = 3;
    data[1] = 5;
    data[2] = 1;
    data[3] = 2;
    data[4] = 5;
    data[5] = 1;

    ASSERT_EQ( *thrust::max_element(
          thrust::make_transform_iterator(data.begin(), thrust::negate<T>()),
          thrust::make_transform_iterator(data.end(),   thrust::negate<T>())), -1);
    ASSERT_EQ( *thrust::max_element(
          thrust::make_transform_iterator(data.begin(), thrust::negate<T>()),
          thrust::make_transform_iterator(data.end(),   thrust::negate<T>()),
          thrust::greater<T>()), -5);
}

TYPED_TEST(MaxElementTests, TestMaxElement)
{
    using Vector = typename TestFixture::input_type;
    using T = typename Vector::value_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<T> h_data = get_random_data<T>(size,
                                                           std::numeric_limits<T>::min(),
                                                           std::numeric_limits<T>::max());
        thrust::device_vector<T> d_data = h_data;

        typename thrust::host_vector<T>::iterator   h_max = thrust::max_element(h_data.begin(), h_data.end());
        typename thrust::device_vector<T>::iterator d_max = thrust::max_element(d_data.begin(), d_data.end());

        ASSERT_EQ(h_max - h_data.begin(), d_max - d_data.begin());

        typename thrust::host_vector<T>::iterator   h_min = thrust::max_element(h_data.begin(), h_data.end(), thrust::less<T>());
        typename thrust::device_vector<T>::iterator d_min = thrust::max_element(d_data.begin(), d_data.end(), thrust::less<T>());

        ASSERT_EQ(h_min - h_data.begin(), d_min - d_data.begin());
    }
}

template<typename ForwardIterator>
ForwardIterator max_element(my_system &system, ForwardIterator first, ForwardIterator)
{
    system.validate_dispatch();
    return first;
}

TEST(MaxElementTests, TestMaxElementDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::max_element(sys, vec.begin(), vec.end());

    ASSERT_EQ(true, sys.is_valid());
}

template<typename ForwardIterator>
ForwardIterator max_element(my_tag, ForwardIterator first, ForwardIterator)
{
    *first = 13;
    return first;
}

TEST(MaxElementTests, TestMaxElementDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::max_element(thrust::retag<my_tag>(vec.begin()),
                        thrust::retag<my_tag>(vec.end()));

    ASSERT_EQ(13, vec.front());
}

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

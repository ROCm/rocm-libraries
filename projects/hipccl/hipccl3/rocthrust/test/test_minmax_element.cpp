/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <thrust/extrema.h>
#include <thrust/iterator/retag.h>

#include "test_header.hpp"

TESTS_DEFINE(MinmaxElementTests, FullTestsParams);
TESTS_DEFINE(MinMaxElementPrimitiveTests, NumericalTestsParams);

TYPED_TEST(MinmaxElementTests, TestMinmaxElementSimple)
{
    using Vector = typename TestFixture::input_type;

    Vector data(6);
    data[0] = 3;
    data[1] = 5;
    data[2] = 1;
    data[3] = 2;
    data[4] = 5;
    data[5] = 1;

    ASSERT_EQ(*thrust::minmax_element(data.begin(), data.end()).first, 1);
    ASSERT_EQ(*thrust::minmax_element(data.begin(), data.end()).second, 5);
    ASSERT_EQ(thrust::minmax_element(data.begin(), data.end()).first - data.begin(), 2);
    ASSERT_EQ(thrust::minmax_element(data.begin(), data.end()).second - data.begin(), 1);
}

TYPED_TEST(MinmaxElementTests, TestMinmaxElementWithTransform)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    // We cannot use unsigned types for this test case
    if(std::is_unsigned<T>::value)
        return;

    Vector data(6);
    data[0] = 3;
    data[1] = 5;
    data[2] = 1;
    data[3] = 2;
    data[4] = 5;
    data[5] = 1;

    ASSERT_EQ(
        *thrust::minmax_element(thrust::make_transform_iterator(data.begin(), thrust::negate<T>()),
                                thrust::make_transform_iterator(data.end(), thrust::negate<T>()))
             .first,
        -5);
    ASSERT_EQ(
        *thrust::minmax_element(thrust::make_transform_iterator(data.begin(), thrust::negate<T>()),
                                thrust::make_transform_iterator(data.end(), thrust::negate<T>()))
             .second,
        -1);
}

TYPED_TEST(MinMaxElementPrimitiveTests, TestMinmaxElement)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<T> h_data = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::device_vector<T> d_data = h_data;

        typename thrust::host_vector<T>::iterator   h_min;
        typename thrust::host_vector<T>::iterator   h_max;
        typename thrust::device_vector<T>::iterator d_min;
        typename thrust::device_vector<T>::iterator d_max;

        h_min = thrust::minmax_element(h_data.begin(), h_data.end()).first;
        d_min = thrust::minmax_element(d_data.begin(), d_data.end()).first;
        h_max = thrust::minmax_element(h_data.begin(), h_data.end()).second;
        d_max = thrust::minmax_element(d_data.begin(), d_data.end()).second;

        ASSERT_EQ(h_min - h_data.begin(), d_min - d_data.begin());
        ASSERT_EQ(h_max - h_data.begin(), d_max - d_data.begin());

        h_max = thrust::minmax_element(h_data.begin(), h_data.end(), thrust::greater<T>()).first;
        d_max = thrust::minmax_element(d_data.begin(), d_data.end(), thrust::greater<T>()).first;
        h_min = thrust::minmax_element(h_data.begin(), h_data.end(), thrust::greater<T>()).second;
        d_min = thrust::minmax_element(d_data.begin(), d_data.end(), thrust::greater<T>()).second;

        ASSERT_EQ(h_min - h_data.begin(), d_min - d_data.begin());
        ASSERT_EQ(h_max - h_data.begin(), d_max - d_data.begin());
    }
}

template <typename ForwardIterator>
thrust::pair<ForwardIterator, ForwardIterator>
    minmax_element(my_system& system, ForwardIterator first, ForwardIterator)
{
    system.validate_dispatch();
    return thrust::make_pair(first, first);
}

TEST(MinmaxElementTests, TestMinmaxElementDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::minmax_element(sys, vec.begin(), vec.end());

    ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator>
thrust::pair<ForwardIterator, ForwardIterator>
    minmax_element(my_tag, ForwardIterator first, ForwardIterator)
{
    *first = 13;
    return thrust::make_pair(first, first);
}

TEST(MinmaxElementTests, TestMinmaxElementDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::minmax_element(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()));

    ASSERT_EQ(13, vec.front());
}
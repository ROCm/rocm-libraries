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

TESTS_DEFINE(MinElementTests, FullTestsParams);
TESTS_DEFINE(MinElementPrimitiveTests, NumericalTestsParams);

TYPED_TEST(MinElementTests, TestMinElementSimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector data(6);
    data[0] = 3;
    data[1] = 5;
    data[2] = 1;
    data[3] = 2;
    data[4] = 5;
    data[5] = 1;

    ASSERT_EQ(*thrust::min_element(data.begin(), data.end()), 1);
    ASSERT_EQ(thrust::min_element(data.begin(), data.end()) - data.begin(), 2);

    ASSERT_EQ(*thrust::min_element(data.begin(), data.end(), thrust::greater<T>()), 5);
    ASSERT_EQ(thrust::min_element(data.begin(), data.end(), thrust::greater<T>()) - data.begin(),
              1);
}

TYPED_TEST(MinElementTests, TestMinElementWithTransform)
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
        *thrust::min_element(thrust::make_transform_iterator(data.begin(), thrust::negate<T>()),
                             thrust::make_transform_iterator(data.end(), thrust::negate<T>())),
        -5);
    ASSERT_EQ(
        *thrust::min_element(thrust::make_transform_iterator(data.begin(), thrust::negate<T>()),
                             thrust::make_transform_iterator(data.end(), thrust::negate<T>()),
                             thrust::greater<T>()),
        -1);
}

TYPED_TEST(MinElementPrimitiveTests, TestMinElement)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<T> h_data = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::device_vector<T> d_data = h_data;

        typename thrust::host_vector<T>::iterator h_min
            = thrust::min_element(h_data.begin(), h_data.end());
        typename thrust::device_vector<T>::iterator d_min
            = thrust::min_element(d_data.begin(), d_data.end());

        ASSERT_EQ(h_data.begin() - h_min, d_data.begin() - d_min);

        typename thrust::host_vector<T>::iterator h_max
            = thrust::min_element(h_data.begin(), h_data.end(), thrust::greater<T>());
        typename thrust::device_vector<T>::iterator d_max
            = thrust::min_element(d_data.begin(), d_data.end(), thrust::greater<T>());

        ASSERT_EQ(h_max - h_data.begin(), d_max - d_data.begin());
    }
}

template <typename ForwardIterator>
ForwardIterator min_element(my_system& system, ForwardIterator first, ForwardIterator)
{
    system.validate_dispatch();
    return first;
}

TEST(MinElementTests, TestMinElementDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::min_element(sys, vec.begin(), vec.end());

    ASSERT_EQ(true, sys.is_valid());
}

template <typename ForwardIterator>
ForwardIterator min_element(my_tag, ForwardIterator first, ForwardIterator)
{
    *first = 13;
    return first;
}

TEST(MinElementTests, TestMinElementDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::min_element(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()));

    ASSERT_EQ(13, vec.front());
}
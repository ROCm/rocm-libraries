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

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/tabulate.h>

#include "test_header.hpp"

TESTS_DEFINE(EqualTests, FullTestsParams);
TESTS_DEFINE(EqualsPrimitiveTests, NumericalTestsParams);

TYPED_TEST(EqualTests, TestEqualSimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector v1(5);
    Vector v2(5);
    v1[0] = T(5);
    v1[1] = T(2);
    v1[2] = T(0);
    v1[3] = T(0);
    v1[4] = T(0);
    v2[0] = T(5);
    v2[1] = T(2);
    v2[2] = T(0);
    v2[3] = T(6);
    v2[4] = T(1);

    ASSERT_EQ(thrust::equal(v1.begin(), v1.end(), v1.begin()), true);
    ASSERT_EQ(thrust::equal(v1.begin(), v1.end(), v2.begin()), false);
    ASSERT_EQ(thrust::equal(v2.begin(), v2.end(), v2.begin()), true);

    ASSERT_EQ(thrust::equal(v1.begin(), v1.begin() + 0, v1.begin()), true);
    ASSERT_EQ(thrust::equal(v1.begin(), v1.begin() + 1, v1.begin()), true);
    ASSERT_EQ(thrust::equal(v1.begin(), v1.begin() + 3, v2.begin()), true);
    ASSERT_EQ(thrust::equal(v1.begin(), v1.begin() + 4, v2.begin()), false);

    ASSERT_EQ(thrust::equal(v1.begin(), v1.end(), v2.begin(), thrust::less_equal<T>()), true);
    ASSERT_EQ(thrust::equal(v1.begin(), v1.end(), v2.begin(), thrust::greater<T>()), false);
}

TYPED_TEST(EqualsPrimitiveTests, TestEqual)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<T> h_data1 = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::host_vector<T> h_data2 = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        thrust::device_vector<T> d_data1 = h_data1;
        thrust::device_vector<T> d_data2 = h_data2;

        //empty ranges
        ASSERT_EQ(thrust::equal(h_data1.begin(), h_data1.begin(), h_data1.begin()), true);
        ASSERT_EQ(thrust::equal(d_data1.begin(), d_data1.begin(), d_data1.begin()), true);

        //symmetric cases
        ASSERT_EQ(thrust::equal(h_data1.begin(), h_data1.end(), h_data1.begin()), true);
        ASSERT_EQ(thrust::equal(d_data1.begin(), d_data1.end(), d_data1.begin()), true);

        if(size > 0)
        {
            h_data1[0] = 0;
            h_data2[0] = 1;
            d_data1[0] = 0;
            d_data2[0] = 1;

            //different vectors
            ASSERT_EQ(thrust::equal(h_data1.begin(), h_data1.end(), h_data2.begin()), false);
            ASSERT_EQ(thrust::equal(d_data1.begin(), d_data1.end(), d_data2.begin()), false);

            //different predicates
            ASSERT_EQ(thrust::equal(
                          h_data1.begin(), h_data1.begin() + 1, h_data2.begin(), thrust::less<T>()),
                      true);
            ASSERT_EQ(thrust::equal(
                          d_data1.begin(), d_data1.begin() + 1, d_data2.begin(), thrust::less<T>()),
                      true);
            ASSERT_EQ(
                thrust::equal(
                    h_data1.begin(), h_data1.begin() + 1, h_data2.begin(), thrust::greater<T>()),
                false);
            ASSERT_EQ(
                thrust::equal(
                    d_data1.begin(), d_data1.begin() + 1, d_data2.begin(), thrust::greater<T>()),
                false);
        }
    }
}

template <typename InputIterator1, typename InputIterator2>
bool equal(my_system& system, InputIterator1, InputIterator1, InputIterator2)
{
    system.validate_dispatch();
    return false;
}

TEST(EqualTests, TestEqualDispatchExplicit)
{
    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::equal(sys, vec.begin(), vec.end(), vec.begin());

    ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator1, typename InputIterator2>
bool equal(my_tag, InputIterator1 first, InputIterator1, InputIterator2)
{
    *first = 13;
    return false;
}

TEST(EqualTests, TestEqualDispatchImplicit)
{
    thrust::device_vector<int> vec(1);

    thrust::equal(thrust::retag<my_tag>(vec.begin()),
                  thrust::retag<my_tag>(vec.end()),
                  thrust::retag<my_tag>(vec.begin()));

    ASSERT_EQ(13, vec.front());
}
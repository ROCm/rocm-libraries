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
#include <thrust/inner_product.h>
#include <thrust/iterator/retag.h>

#include "test_header.hpp"

TESTS_DEFINE(InnerProductTests, FullTestsParams);
TESTS_DEFINE(PrimitiveInnerProductTests, NumericalTestsParams);

template <class T>
T clip_infinity(T val)
{
    T min = std::numeric_limits<T>::min();
    T max = std::numeric_limits<T>::max();
    if(val > max)
        return max;
    if(val < min)
        return min;
    return val;
}

TEST(InnerProductTests, UsingHip)
{
    ASSERT_EQ(THRUST_DEVICE_SYSTEM, THRUST_DEVICE_SYSTEM_HIP);
}

TYPED_TEST(InnerProductTests, InnerProductSimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector v1(3);
    Vector v2(3);
    v1[0] = 1;
    v1[1] = -2;
    v1[2] = 3;
    v2[0] = -4;
    v2[1] = 5;
    v2[2] = 6;

    T init   = 3;
    T result = thrust::inner_product(v1.begin(), v1.end(), v2.begin(), init);

    ASSERT_NEAR(result, (T)7, (T)0.01);
}

template <typename InputIterator1, typename InputIterator2, typename OutputType>
int inner_product(my_system& system, InputIterator1, InputIterator1, InputIterator2, OutputType)
{
    system.validate_dispatch();
    return 13;
}

TEST(InnerProductTests, InnerProductDispatchExplicit)
{
    thrust::device_vector<int> vec;

    my_system sys(0);
    thrust::inner_product(sys, vec.begin(), vec.end(), vec.begin(), 0);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator1, typename InputIterator2, typename OutputType>
int inner_product(my_tag, InputIterator1, InputIterator1, InputIterator2, OutputType)
{
    return 13;
}

TEST(InnerProductTests, InnerProductDispatchImplicit)
{
    thrust::device_vector<int> vec;

    int result = thrust::inner_product(thrust::retag<my_tag>(vec.begin()),
                                       thrust::retag<my_tag>(vec.end()),
                                       thrust::retag<my_tag>(vec.begin()),
                                       0);

    ASSERT_EQ(13, result);
}

TYPED_TEST(InnerProductTests, InnerProductWithOperator)
{
    using Vector   = typename TestFixture::input_type;
    using T        = typename Vector::value_type;
    T error_margin = (T)0.01;

    Vector v1(3);
    Vector v2(3);
    v1[0] = 1;
    v1[1] = -2;
    v1[2] = 3;
    v2[0] = -1;
    v2[1] = 3;
    v2[2] = 6;

    // compute (v1 - v2) and perform a multiplies reduction
    T init   = 3;
    T result = thrust::inner_product(
        v1.begin(), v1.end(), v2.begin(), init, thrust::multiplies<T>(), thrust::minus<T>());
    ASSERT_NEAR(result, (T)90, error_margin);
}

TYPED_TEST(PrimitiveInnerProductTests, InnerProductWithRandomData)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        T error_margin = (T)0.01 * size;
        T min          = (T)std::numeric_limits<T>::min() / (size + 1);
        T max          = (T)std::numeric_limits<T>::max() / (size + 1);

        thrust::host_vector<T> h_v1 = get_random_data<T>(size, min, max);
        thrust::host_vector<T> h_v2 = get_random_data<T>(size, min, max);

        thrust::device_vector<T> d_v1 = h_v1;
        thrust::device_vector<T> d_v2 = h_v2;

        T init = 13;

        T expected = thrust::inner_product(h_v1.begin(), h_v1.end(), h_v2.begin(), init);
        T result   = thrust::inner_product(d_v1.begin(), d_v1.end(), d_v2.begin(), init);

        ASSERT_NEAR(clip_infinity(expected), clip_infinity(result), error_margin);
    }
};

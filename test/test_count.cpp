/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/tabulate.h>

#include "test_header.hpp"

TESTS_DEFINE(CountTests, FullTestsParams);
TESTS_DEFINE(CountPrimitiveTests, NumericalTestsParams);

TYPED_TEST(CountTests, TestCountSimple)
{
    using Vector = typename TestFixture::input_type;
    using Policy = typename TestFixture::execution_policy;
    using T      = typename Vector::value_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    Vector data(5);
    data[0] = T(1);
    data[1] = T(1);
    data[2] = T(0);
    data[3] = T(0);
    data[4] = T(1);

    ASSERT_EQ(thrust::count(Policy{}, data.begin(), data.end(), T(0)), 2);
    ASSERT_EQ(thrust::count(Policy{}, data.begin(), data.end(), T(1)), 3);
    ASSERT_EQ(thrust::count(Policy{}, data.begin(), data.end(), T(2)), 0);
}

TYPED_TEST(CountPrimitiveTests, TestCount)
{
    using T = typename TestFixture::input_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size= " << size);

        for(auto seed : get_seeds())
        {
            SCOPED_TRACE(testing::Message() << "with seed= " << seed);

            thrust::host_vector<T> h_data = get_random_data<T>(
                size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
            thrust::device_vector<T> d_data = h_data;

            size_t cpu_result = thrust::count(h_data.begin(), h_data.end(), T(5));
            size_t gpu_result = thrust::count(d_data.begin(), d_data.end(), T(5));

            ASSERT_EQ(cpu_result, gpu_result);
        }
    }
}

template <typename T>
struct greater_than_five
{
    __host__ __device__ bool operator()(const T& x) const
    {
        return x > 5;
    }
};

TYPED_TEST(CountTests, TestCountIfSimple)
{
    using Vector = typename TestFixture::input_type;
    using Policy = typename TestFixture::execution_policy;
    using T      = typename Vector::value_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    Vector data(5);
    data[0] = 1;
    data[1] = 6;
    data[2] = 1;
    data[3] = 9;
    data[4] = 2;

    ASSERT_EQ(thrust::count_if(Policy{}, data.begin(), data.end(), greater_than_five<T>()), 2);
}

TYPED_TEST(CountTests, TestCountIf)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    for(auto size : get_sizes())
    {
        SCOPED_TRACE(testing::Message() << "with size= " << size);

        for(auto seed : get_seeds())
        {
            SCOPED_TRACE(testing::Message() << "with seed= " << seed);

            thrust::host_vector<T> h_data = get_random_data<T>(
                size, get_default_limits<T>::min(), get_default_limits<T>::max(), seed);
            thrust::device_vector<T> d_data = h_data;
            size_t                   cpu_result
                = thrust::count_if(h_data.begin(), h_data.end(), greater_than_five<T>());
            size_t gpu_result
                = thrust::count_if(d_data.begin(), d_data.end(), greater_than_five<T>());

            ASSERT_EQ(cpu_result, gpu_result);
        }
    }
}

TYPED_TEST(CountTests, TestCountFromConstIteratorSimple)
{
    using Vector = typename TestFixture::input_type;
    using Policy = typename TestFixture::execution_policy;
    using T      = typename Vector::value_type;

    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    Vector data(5);
    data[0] = T(1);
    data[1] = T(1);
    data[2] = T(0);
    data[3] = T(0);
    data[4] = T(1);

    ASSERT_EQ(thrust::count(Policy{}, data.cbegin(), data.cend(), T(0)), 2);
    ASSERT_EQ(thrust::count(Policy{}, data.cbegin(), data.cend(), T(1)), 3);
    ASSERT_EQ(thrust::count(Policy{}, data.cbegin(), data.cend(), T(2)), 0);
}

template <typename InputIterator, typename EqualityComparable>
int count(my_system& system, InputIterator, InputIterator, EqualityComparable x)
{
    system.validate_dispatch();
    return x;
}

TEST(CountTests, TestCountDispatchExplicit)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_vector<int> vec(1);

    my_system sys(0);
    thrust::count(sys, vec.begin(), vec.end(), 13);

    ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator, typename EqualityComparable>
int count(my_tag, InputIterator, InputIterator, EqualityComparable x)
{
    return x;
}

TEST(CountTests, TestCountDispatchImplicit)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
    
    thrust::device_vector<int> vec(1);

    int result
        = thrust::count(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), 13);

    ASSERT_EQ(13, result);
}

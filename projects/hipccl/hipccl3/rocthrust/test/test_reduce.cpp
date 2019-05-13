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

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/reduce.h>

#include "test_header.hpp"

TESTS_DEFINE(ReduceTests, FullTestsParams);
TESTS_DEFINE(ReduceIntegerTests, UnsignedIntegerTestsParams);
TESTS_DEFINE(ReducePrimitiveTests, NumericalTestsParams);

template <typename T>
struct plus_mod_10
{
    __host__ __device__ T operator()(T rhs, T lhs) const
    {
        return ((lhs % 10) + (rhs % 10)) % 10;
    }
};

TYPED_TEST(ReduceTests, TestReduceSimple)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector v(3);
    v[0] = 1;
    v[1] = -2;
    v[2] = 3;

    // no initializer
    ASSERT_EQ(thrust::reduce(v.begin(), v.end()), 2);

    // with initializer
    ASSERT_EQ(thrust::reduce(v.begin(), v.end(), T(10)), 12);
}

template <typename InputIterator>
int reduce(my_system& system, InputIterator, InputIterator)
{
    system.validate_dispatch();
    return 13;
}

TEST(ReduceTests, TestReduceDispatchExplicit)
{
    thrust::device_vector<int> vec;

    my_system sys(0);
    thrust::reduce(sys, vec.begin(), vec.end());

    ASSERT_EQ(true, sys.is_valid());
}

template <typename InputIterator>
int reduce(my_tag, InputIterator, InputIterator)
{
    return 13;
}

TEST(ReduceTests, TestReduceDispatchImplicit)
{
    thrust::device_vector<int> vec;

    my_system sys(0);
    thrust::reduce(sys, vec.begin(), vec.end());

    ASSERT_EQ(true, sys.is_valid());
}

TYPED_TEST(ReducePrimitiveTests, TestReduce)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<T> h_data = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        thrust::device_vector<T> d_data = h_data;

        T init = T(13);

        T h_result = thrust::reduce(h_data.begin(), h_data.end(), init);
        T d_result = thrust::reduce(d_data.begin(), d_data.end(), init);

        ASSERT_EQ(h_result, d_result);
    }
}

TYPED_TEST(ReduceTests, TestReduceMixedTypes)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    if(std::is_floating_point<T>::value)
    {
        Vector float_input(4);
        float_input[0] = T(1.5);
        float_input[1] = T(2.5);
        float_input[2] = T(3.5);
        float_input[3] = T(4.5);

        // float -> int should use using plus<int> operator by default
        ASSERT_EQ(thrust::reduce(float_input.begin(), float_input.end(), (int)0), 10);
    }
    else
    {
        Vector int_input(4);
        int_input[0] = T(1);
        int_input[1] = T(2);
        int_input[2] = T(3);
        int_input[3] = T(4);

        // int -> float should use using plus<float> operator by default
        ASSERT_EQ(thrust::reduce(int_input.begin(), int_input.end(), (float)0.5), 10.5);
    }
}

TYPED_TEST(ReduceIntegerTests, TestReduceWithOperator)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<T> h_data = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        thrust::device_vector<T> d_data = h_data;

        T init = T(3);

        T cpu_result = thrust::reduce(h_data.begin(), h_data.end(), init, plus_mod_10<T>());
        T gpu_result = thrust::reduce(d_data.begin(), d_data.end(), init, plus_mod_10<T>());

        ASSERT_EQ(cpu_result, gpu_result);
    }
}

template <typename T>
struct plus_mod3
{
    T* table;

    plus_mod3(T* table)
        : table(table)
    {
    }

    __host__ __device__ T operator()(T a, T b)
    {
        return table[(int)(a + b)];
    }
};

TYPED_TEST(ReduceTests, TestReduceWithIndirection)
{
    using Vector = typename TestFixture::input_type;
    using T      = typename Vector::value_type;

    Vector data(7);
    data[0] = 0;
    data[1] = 1;
    data[2] = 2;
    data[3] = 1;
    data[4] = 2;
    data[5] = 0;
    data[6] = 1;

    Vector table(6);
    table[0] = 0;
    table[1] = 1;
    table[2] = 2;
    table[3] = 0;
    table[4] = 1;
    table[5] = 2;

    T result = thrust::reduce(
        data.begin(), data.end(), T(0), plus_mod3<T>(thrust::raw_pointer_cast(&table[0])));

    ASSERT_EQ(result, T(1));
}

TYPED_TEST(ReducePrimitiveTests, TestReduceCountingIterator)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        size_t n = thrust::min<size_t>(size, std::numeric_limits<T>::max());

        thrust::counting_iterator<T, thrust::host_system_tag> h_first
            = thrust::make_counting_iterator<T>(0);
        thrust::counting_iterator<T, thrust::device_system_tag> d_first
            = thrust::make_counting_iterator<T>(0);

        T init = 13;

        T h_result = thrust::reduce(h_first, h_first + n, init);
        T d_result = thrust::reduce(d_first, d_first + n, init);

        if(std::is_floating_point<T>::value)
        {
            ASSERT_NEAR(h_result, d_result, h_result * 0.01);
        }
        else
        {
            ASSERT_EQ(h_result, d_result);
        }
    }
}
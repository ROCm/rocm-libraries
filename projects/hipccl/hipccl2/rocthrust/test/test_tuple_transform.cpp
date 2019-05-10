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

#include <thrust/transform.h>
#include <thrust/tuple.h>

#include "test_header.hpp"

TESTS_DEFINE(TupleTransformTests, SignedIntegerTestsParams);

struct MakeTupleFunctor
{
    template <typename T1, typename T2>
    __host__ __device__ thrust::tuple<T1, T2> operator()(T1& lhs, T2& rhs)
    {
        return thrust::make_tuple(lhs, rhs);
    }
};

template <int N>
struct GetFunctor
{
    template <typename Tuple>
    __host__ __device__
        typename thrust::access_traits<typename thrust::tuple_element<N, Tuple>::type>::const_type
        operator()(const Tuple& t)
    {
        return thrust::get<N>(t);
    }
};

TYPED_TEST(TupleTransformTests, TestTupleTransform)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<T> h_t1 = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        thrust::host_vector<T> h_t2 = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        // zip up the data
        thrust::host_vector<thrust::tuple<T, T>> h_tuples(size);
        thrust::transform(
            h_t1.begin(), h_t1.end(), h_t2.begin(), h_tuples.begin(), MakeTupleFunctor());

        // copy to device
        thrust::device_vector<thrust::tuple<T, T>> d_tuples = h_tuples;

        thrust::device_vector<T> d_t1(size), d_t2(size);

        // select 0th
        thrust::transform(d_tuples.begin(), d_tuples.end(), d_t1.begin(), GetFunctor<0>());

        // select 1st
        thrust::transform(d_tuples.begin(), d_tuples.end(), d_t2.begin(), GetFunctor<1>());

        ASSERT_EQ(h_t1, d_t1);
        ASSERT_EQ(h_t2, d_t2);

        ASSERT_EQ_QUIET(h_tuples, d_tuples);
    }
}

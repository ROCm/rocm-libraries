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

#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include "test_header.hpp"

TESTS_DEFINE(TupleReduceTests, IntegerTestsParams);

struct SumTupleFunctor
{
    template <typename Tuple>
    __host__ __device__ Tuple operator()(const Tuple& lhs, const Tuple& rhs)
    {
        using thrust::get;

        return thrust::make_tuple(get<0>(lhs) + get<0>(rhs), get<1>(lhs) + get<1>(rhs));
    }
};

struct MakeTupleFunctor
{
    template <typename T1, typename T2>
    __host__ __device__ thrust::tuple<T1, T2> operator()(T1& lhs, T2& rhs)
    {
        return thrust::make_tuple(lhs, rhs);
    }
};

TYPED_TEST(TupleReduceTests, TestTupleReduce)
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

        thrust::tuple<T, T> zero(0, 0);

        // sum on host
        thrust::tuple<T, T> h_result
            = thrust::reduce(h_tuples.begin(), h_tuples.end(), zero, SumTupleFunctor());

        // sum on device
        thrust::tuple<T, T> d_result
            = thrust::reduce(d_tuples.begin(), d_tuples.end(), zero, SumTupleFunctor());

        ASSERT_EQ_QUIET(h_result, d_result);
    }
}

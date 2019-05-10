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

TESTS_DEFINE(TupleSortTests, IntegerTestsParams);

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

TYPED_TEST(TupleSortTests, TestTupleStableSort)
{
    using T = typename TestFixture::input_type;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<T> h_keys = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        thrust::host_vector<T> h_values = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        thrust::host_vector<thrust::tuple<T, T>> h_tuples(size);
        transform(
            h_keys.begin(), h_keys.end(), h_values.begin(), h_tuples.begin(), MakeTupleFunctor());

        // copy to device
        thrust::device_vector<thrust::tuple<T, T>> d_tuples = h_tuples;

        // sort on host
        thrust::stable_sort(h_tuples.begin(), h_tuples.end());

        // sort on device
        thrust::stable_sort(d_tuples.begin(), d_tuples.end());

        ASSERT_EQ(true, is_sorted(d_tuples.begin(), d_tuples.end()));

        // select keys
        thrust::transform(h_tuples.begin(), h_tuples.end(), h_keys.begin(), GetFunctor<0>());

        thrust::device_vector<T> d_keys(h_keys.size());
        thrust::transform(d_tuples.begin(), d_tuples.end(), d_keys.begin(), GetFunctor<0>());

        // select values
        thrust::transform(h_tuples.begin(), h_tuples.end(), h_values.begin(), GetFunctor<1>());

        thrust::device_vector<T> d_values(h_values.size());
        thrust::transform(d_tuples.begin(), d_tuples.end(), d_values.begin(), GetFunctor<1>());

        ASSERT_EQ(h_keys, d_keys);
        ASSERT_EQ(h_values, d_values);
    }
}

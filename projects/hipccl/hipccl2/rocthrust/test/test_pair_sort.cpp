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

#include <thrust/pair.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "test_header.hpp"

TESTS_DEFINE(PairSortTests, NumericalTestsParams);

struct make_pair_functor
{
    template <typename T1, typename T2>
    __host__ __device__ thrust::pair<T1, T2> operator()(const T1& x, const T2& y)
    {
        return thrust::make_pair(x, y);
    } // end operator()()
}; // end make_pair_functor

TYPED_TEST(PairSortTests, TestPairStableSortByKey)
{
    using T = typename TestFixture::input_type;
    using P = thrust::pair<T, T>;

    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        thrust::host_vector<T> h_p1 = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        ;

        thrust::host_vector<T> h_p2 = get_random_data<T>(
            size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
        ;

        thrust::host_vector<P> h_pairs(size);

        thrust::host_vector<int> h_values(size);
        thrust::sequence(h_values.begin(), h_values.end());

        // zip up pairs on the host
        thrust::transform(
            h_p1.begin(), h_p1.end(), h_p2.begin(), h_pairs.begin(), make_pair_functor());

        // device arrays
        thrust::device_vector<P>   d_pairs  = h_pairs;
        thrust::device_vector<int> d_values = h_values;

        // sort on the host
        thrust::stable_sort_by_key(h_pairs.begin(), h_pairs.end(), h_values.begin());

        // sort on the device
        thrust::stable_sort_by_key(d_pairs.begin(), d_pairs.end(), d_values.begin());

        ASSERT_EQ_QUIET(h_pairs, d_pairs);
        ASSERT_EQ_QUIET(h_values, d_values);
    }
}

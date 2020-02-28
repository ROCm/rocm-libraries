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

#include <thrust/functional.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>

#include "test_header.hpp"

TESTS_DEFINE(SetIntersectionKeyValueTests, NumericalTestsParams);

TYPED_TEST(SetIntersectionKeyValueTests, TestSetIntersectionKeyValue)
{
    using U = typename TestFixture::input_type;
    using T = key_value<U, U>;

    const std::vector<size_t> sizes = get_sizes();

    for(auto size : sizes)
    {
        SCOPED_TRACE(testing::Message() << "with size= " << size);
        for(size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
        {
            unsigned int seed_value
                = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed= " << seed_value);

            thrust::host_vector<U> h_keys_a = get_random_data<U>(
                size, std::numeric_limits<U>::min(), std::numeric_limits<U>::max(), seed_value);
            thrust::host_vector<U> h_values_a = get_random_data<U>(
                size,
                std::numeric_limits<U>::min(),
                std::numeric_limits<U>::max(),
                seed_value + seed_value_addition
            );

            thrust::host_vector<U> h_keys_b = get_random_data<U>(
                size,
                std::numeric_limits<U>::min(),
                std::numeric_limits<U>::max(),
                seed_value + 2 * seed_value_addition
            );
            thrust::host_vector<U> h_values_b = get_random_data<U>(
                size,
                std::numeric_limits<U>::min(),
                std::numeric_limits<U>::max(),
                seed_value + 3 * seed_value_addition
            );

            thrust::host_vector<T> h_a(size), h_b(size);

            for(size_t i = 0; i < size; ++i)
            {
                h_a[i] = T(h_keys_a[i], h_values_a[i]);
                h_b[i] = T(h_keys_b[i], h_values_b[i]);
            }

            thrust::stable_sort(h_a.begin(), h_a.end());
            thrust::stable_sort(h_b.begin(), h_b.end());

            thrust::device_vector<T> d_a = h_a;
            thrust::device_vector<T> d_b = h_b;

            thrust::host_vector<T>   h_result(size);
            thrust::device_vector<T> d_result(size);

            typename thrust::host_vector<T>::iterator   h_end;
            typename thrust::device_vector<T>::iterator d_end;

            h_end = thrust::set_intersection(
                h_a.begin(), h_a.end(), h_b.begin(), h_b.end(), h_result.begin());
            h_result.resize(h_end - h_result.begin());

            d_end = thrust::set_intersection(
                d_a.begin(), d_a.end(), d_b.begin(), d_b.end(), d_result.begin());

            d_result.resize(d_end - d_result.begin());

            thrust::host_vector<T> d_result_host(d_result);

            ASSERT_EQ(h_result, d_result_host);
        }
    }
}

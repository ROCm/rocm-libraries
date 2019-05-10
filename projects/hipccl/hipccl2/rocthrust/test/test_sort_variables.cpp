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

#include "test_header.hpp"

TESTS_DEFINE(SortByKeyVariableTests, AllIntegerTestsParams);

TYPED_TEST(SortByKeyVariableTests, TestSortVariableBits)
{
    using T = typename TestFixture::input_type;

    for(auto size : get_sizes())
    {
        for(size_t num_bits = 0; num_bits < 8 * sizeof(T); num_bits += 3)
        {
            SCOPED_TRACE(testing::Message() << "with size = " << size);

            thrust::host_vector<T> h_keys = get_random_data<T>(
                size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

            size_t mask = (1 << num_bits) - 1;
            for(size_t i = 0; i < size; i++)
                h_keys[i] &= mask;

            thrust::host_vector<T>   reference = h_keys;
            thrust::device_vector<T> d_keys    = h_keys;

            std::sort(reference.begin(), reference.end());

            thrust::sort(h_keys.begin(), h_keys.end());
            thrust::sort(d_keys.begin(), d_keys.end());

            ASSERT_EQ(reference, h_keys);
            ASSERT_EQ(h_keys, d_keys);
        }
    }
}
